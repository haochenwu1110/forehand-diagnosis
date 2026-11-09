"""先用一个很短的请求检查 API key 和网络是否可用（每个模型只花几个 token）。

    python check_api.py --config generators.json                # 检查配置里的全部模型
    python check_api.py --config generators.json --only deepseek
"""
import argparse
import os

from llm_clients import LLMError, load_model_configs, make_client, parse_json_reply

HINT = {401: "key 错误或未生效", 403: "无权限，或所在地区不支持（Gemini 在中国大陆需代理）",
        402: "账户余额不足", 404: "模型 ID 不存在，请到控制台核对", 429: "超出频率 / 免费额度，稍后再试",
        400: "请求参数不被接受，把完整报错发给我"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="generators.json")
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    print(f"代理：{proxy or '未设置'}")
    for c in load_model_configs(a.config, a.only):
        c = dict(c, max_retries=1, timeout=60)
        try:
            r = make_client(c).chat("只输出 JSON。", '请输出 {"ok": true}', max_tokens=2000)
            ok = parse_json_reply(r.text).get("ok") is True
            print(f"✓ {c['name']:<10} 返回模型 {r.model}，{r.latency_s}s，回复{'正常' if ok else '格式异常：' + r.text[:80]}")
        except LLMError as e:
            hint = "先设置这个环境变量（export 或 .env）" if "未设置环境变量" in str(e) else \
                HINT.get(e.status, "网络不通：检查网络或代理 HTTPS_PROXY")
            print(f"✗ {c['name']:<10} {e}\n    → {hint}")
        except Exception as e:  # noqa: BLE001
            print(f"✗ {c['name']:<10} {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
