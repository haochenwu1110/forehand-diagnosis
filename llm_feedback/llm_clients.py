"""统一的大模型调用层：Claude（Anthropic）、OpenAI、Google Gemini，以及任何 OpenAI 兼容接口
（DeepSeek、通义千问 DashScope、智谱 GLM、Moonshot/Kimi、OpenRouter、本地 vLLM / Ollama 等）。

只用 Python 标准库（urllib），不依赖各家 SDK，便于复现和长期存档。API key 只从环境变量读取，不写入任何日志。

    from llm_clients import make_client
    c = make_client({"provider": "anthropic", "model": "claude-opus-5-5"})
    r = c.chat(system="...", user="...", temperature=0, max_tokens=4000)
    r.text, r.model, r.usage, r.latency_s

模型配置字段（judges.example.json / generators.example.json 中的一项）：
    name          自定义名称，用作输出文件名（默认 = model）
    provider      anthropic | openai | gemini | openai_compatible | mock
    model         控制台里的正式模型 ID
    temperature   数值；null 表示不传（部分推理模型只接受默认温度）
    max_tokens    最大输出 token 数
    base_url      仅 openai_compatible 需要，如 https://api.deepseek.com
    api_key_env   存放 key 的环境变量名（各 provider 有默认值，见 DEFAULT_KEY_ENV）
    json_mode     true 时请求 JSON 输出（OpenAI response_format / Gemini responseMimeType）；接口不支持时自动关闭重试
    extra         原样合并进请求体的额外字段（如 {"reasoning_effort": "low"}）
"""
import json
import os
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

DEFAULT_KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "openai_compatible": "OPENAI_COMPATIBLE_API_KEY",
}
ANTHROPIC_VERSION = "2023-06-01"
RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}


@dataclass
class LLMResponse:
    text: str
    model: str                      # 服务端实际返回的模型名（写进日志，论文据此报告）
    usage: dict = field(default_factory=dict)
    latency_s: float = 0.0
    stop_reason: str = ""
    raw: dict = field(default_factory=dict)


class LLMError(RuntimeError):
    def __init__(self, msg, status=None, body=""):
        super().__init__(msg)
        self.status, self.body = status, body


def _post(url, headers, body, timeout=180):
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), method="POST",
                                 headers={"content-type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        txt = e.read().decode("utf-8", "replace")
        raise LLMError(f"HTTP {e.code}: {txt[:500]}", status=e.code, body=txt) from None
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        raise LLMError(f"网络错误: {e}", status=None) from None


class BaseClient:
    provider = "base"

    def __init__(self, cfg):
        self.cfg = dict(cfg)
        self.model = cfg["model"]
        self.name = cfg.get("name") or self.model
        self.max_retries = int(cfg.get("max_retries", 5))
        self.timeout = float(cfg.get("timeout", 180))

    def _key(self):
        env = self.cfg.get("api_key_env") or DEFAULT_KEY_ENV.get(self.provider)
        k = os.environ.get(env or "", "")
        if not k:
            raise LLMError(f"[{self.name}] 未设置环境变量 {env}")
        return k

    def chat(self, system, user, temperature=None, max_tokens=None):
        """带指数退避重试。temperature / max_tokens 不给时用模型配置里的值。"""
        temperature = self.cfg.get("temperature") if temperature is None else temperature
        max_tokens = max_tokens or int(self.cfg.get("max_tokens", 4000))
        delay = 2.0
        for attempt in range(self.max_retries + 1):
            t0 = time.time()
            try:
                r = self._chat(system, user, temperature, max_tokens)
                r.latency_s = round(time.time() - t0, 2)
                return r
            except LLMError as e:
                if attempt >= self.max_retries or (e.status is not None and e.status not in RETRY_STATUS):
                    raise
                time.sleep(delay + random.random())
                delay = min(delay * 2, 60)

    def _chat(self, system, user, temperature, max_tokens):
        raise NotImplementedError


class AnthropicClient(BaseClient):
    provider = "anthropic"

    def _chat(self, system, user, temperature, max_tokens):
        body = dict(model=self.model, max_tokens=max_tokens, system=system,
                    messages=[{"role": "user", "content": user}], **self.cfg.get("extra", {}))
        if temperature is not None:
            body["temperature"] = temperature
        url = self.cfg.get("base_url", "https://api.anthropic.com").rstrip("/") + "/v1/messages"
        d = _post(url, {"x-api-key": self._key(), "anthropic-version": ANTHROPIC_VERSION}, body, self.timeout)
        text = "".join(b.get("text", "") for b in d.get("content", []) if b.get("type") == "text")
        return LLMResponse(text, d.get("model", self.model), d.get("usage", {}), stop_reason=d.get("stop_reason", ""),
                           raw={k: v for k, v in d.items() if k != "content"})


class OpenAIClient(BaseClient):
    """OpenAI Chat Completions；openai_compatible 只是换 base_url 和 key。"""
    provider = "openai"
    default_base = "https://api.openai.com/v1"
    token_field = "max_completion_tokens"

    def _chat(self, system, user, temperature, max_tokens):
        body = dict(model=self.model, messages=[{"role": "system", "content": system},
                                                {"role": "user", "content": user}],
                    **{self.token_field: max_tokens}, **self.cfg.get("extra", {}))
        if temperature is not None and not self.cfg.get("_drop_temperature"):
            body["temperature"] = temperature
        if self.cfg.get("json_mode") and not self.cfg.get("_drop_json_mode"):
            body["response_format"] = {"type": "json_object"}
        url = (self.cfg.get("base_url") or self.default_base).rstrip("/") + "/chat/completions"
        try:
            d = _post(url, {"authorization": f"Bearer {self._key()}"}, body, self.timeout)
        except LLMError as e:
            # 部分推理模型不接受 temperature / response_format / max_completion_tokens：自动去掉后重试一次，并记在配置里
            low = (e.body or "").lower()
            if e.status == 400 and "temperature" in low and "_drop_temperature" not in self.cfg:
                self.cfg["_drop_temperature"] = True
                return self._chat(system, user, temperature, max_tokens)
            if e.status == 400 and "response_format" in low and "_drop_json_mode" not in self.cfg:
                self.cfg["_drop_json_mode"] = True
                return self._chat(system, user, temperature, max_tokens)
            if e.status == 400 and "max_completion_tokens" in low and self.token_field != "max_tokens":
                self.token_field = "max_tokens"
                return self._chat(system, user, temperature, max_tokens)
            raise
        ch = d["choices"][0]
        return LLMResponse(ch["message"].get("content") or "", d.get("model", self.model), d.get("usage", {}),
                           stop_reason=ch.get("finish_reason", ""), raw={k: v for k, v in d.items() if k != "choices"})


class OpenAICompatibleClient(OpenAIClient):
    provider = "openai_compatible"
    token_field = "max_tokens"          # 多数兼容接口仍用旧字段名

    def __init__(self, cfg):
        super().__init__(cfg)
        if not cfg.get("base_url"):
            raise LLMError(f"[{self.name}] openai_compatible 需要 base_url")


class GeminiClient(BaseClient):
    provider = "gemini"

    def _chat(self, system, user, temperature, max_tokens):
        gen = {"maxOutputTokens": max_tokens}
        if temperature is not None:
            gen["temperature"] = temperature
        if self.cfg.get("json_mode") and not self.cfg.get("_drop_json_mode"):
            gen["responseMimeType"] = "application/json"
        body = {"systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": gen, **self.cfg.get("extra", {})}
        base = self.cfg.get("base_url", "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
        d = _post(f"{base}/models/{self.model}:generateContent", {"x-goog-api-key": self._key()}, body, self.timeout)
        cands = d.get("candidates") or []
        if not cands:
            raise LLMError(f"[{self.name}] 无候选输出：{json.dumps(d.get('promptFeedback', {}), ensure_ascii=False)}",
                           status=200)
        c0 = cands[0]
        text = "".join(p.get("text", "") for p in c0.get("content", {}).get("parts", []) if not p.get("thought"))
        return LLMResponse(text, d.get("modelVersion", self.model), d.get("usageMetadata", {}),
                           stop_reason=c0.get("finishReason", ""), raw={"modelVersion": d.get("modelVersion")})


class MockClient(BaseClient):
    """离线测试用：不联网，用 cfg["responder"]（函数）根据 system / user 生成回复。结果不得用于论文。"""
    provider = "mock"

    def _chat(self, system, user, temperature, max_tokens):
        fn = self.cfg.get("responder")
        if fn is None:
            raise LLMError("mock 客户端需要 responder", status=400)
        return LLMResponse(fn(system, user, self.cfg), f"mock:{self.model}", {"input_tokens": 0, "output_tokens": 0})


PROVIDERS = {"anthropic": AnthropicClient, "openai": OpenAIClient, "gemini": GeminiClient,
             "openai_compatible": OpenAICompatibleClient, "mock": MockClient}


def make_client(cfg):
    p = cfg.get("provider")
    if p not in PROVIDERS:
        raise ValueError(f"未知 provider: {p}（可选 {sorted(PROVIDERS)}）")
    return PROVIDERS[p](cfg)


def parse_json_reply(text):
    """去掉 ```json 围栏和前后说明文字，解析第一个完整 JSON 对象。"""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else ""
        t = t.rsplit("```", 1)[0]
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        i = t.find("{")
        if i < 0:
            raise
        obj, _ = json.JSONDecoder().raw_decode(t[i:])
        return obj


def load_model_configs(path, only=None):
    cfgs = json.load(open(path, encoding="utf-8"))
    cfgs = cfgs["models"] if isinstance(cfgs, dict) else cfgs
    cfgs = [c for c in cfgs if not c.get("disabled")]
    for c in cfgs:
        c.setdefault("name", c["model"])
    if only:
        keep = set(only.split(","))
        cfgs = [c for c in cfgs if c["name"] in keep]
    names = [c["name"] for c in cfgs]
    if len(set(names)) != len(names):
        raise ValueError(f"模型配置 name 重复：{names}")
    return cfgs
