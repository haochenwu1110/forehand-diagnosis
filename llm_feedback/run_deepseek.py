"""一键运行：用 DeepSeek 判定论文里 Claude 生成的纠正建议是否忠实于诊断。

    python run_deepseek.py

会依次做三件事：
  1. 用几个 token 测试 key 和网络；
  2. 在示例输出里植入 10 处已知错误，作为负对照（检验 DeepSeek 能不能发现错误）；
  3. 让 DeepSeek 逐名运动员判定"正式输出"和"负对照"，共 12 次调用，汇总成报告。
结果在 judge_deepseek/ 文件夹，并自动打包成 judge_deepseek_结果.zip —— 把这个 zip 发回即可。
"""
import getpass
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.chdir(HERE)
sys.path.insert(0, str(HERE))

import make_negative_control as NC  # noqa: E402
import run_judges  # noqa: E402
from llm_clients import LLMError, load_model_configs, make_client, parse_json_reply  # noqa: E402

MOCK = "--mock" in sys.argv          # 仅用于离线测试流程
OUT = HERE / ("judge_mock" if MOCK else "judge_deepseek")
TARGET = HERE / "outputs_example_chat.json"   # 论文表 7 和表 S5a–S5d 中的 Claude 输出


def step(n, text):
    print(f"\n========== 第 {n} 步：{text} ==========")


def main():
    if not MOCK:
        if not os.environ.get("DEEPSEEK_API_KEY"):
            os.environ["DEEPSEEK_API_KEY"] = getpass.getpass("请粘贴 DeepSeek API key（输入时不显示，粘贴后按回车）：").strip()
        step(1, "测试 key 和网络")
        cfg = dict(load_model_configs(HERE / "judges_deepseek.json")[0], max_retries=1, timeout=60)
        try:
            r = make_client(cfg).chat("只输出 JSON。", '请输出 {"ok": true}', max_tokens=2000)
            parse_json_reply(r.text)
            print(f"✓ 连接成功，返回模型：{r.model}")
        except Exception as e:  # noqa: BLE001
            hint = {401: "key 不对，请重新复制", 402: "余额不足，请先在 platform.deepseek.com 充值",
                    404: "模型名不对", 429: "请求太频繁，稍等再试"}.get(getattr(e, "status", None), "检查网络")
            print(f"✗ 连接失败：{e}\n  → {hint}")
            return

    step(2, "生成负对照（植入 10 处已知错误）")
    inp = json.loads((HERE / "inputs.json").read_text(encoding="utf-8"))
    out = json.loads(TARGET.read_text(encoding="utf-8"))
    seeded, seeds = NC.seed(inp, out)
    nc = HERE / "negative_control"
    nc.mkdir(exist_ok=True)
    (nc / "outputs_seeded.json").write_text(json.dumps(seeded, ensure_ascii=False, indent=1), encoding="utf-8")
    (nc / "seeds.json").write_text(json.dumps(dict(source=str(TARGET), seeds=seeds), ensure_ascii=False, indent=1),
                                   encoding="utf-8")
    for s in seeds:
        print(f"  {s['id']} {s['player']} 第 {s['rank']} 条：{s['desc']}")

    step(3, "DeepSeek 逐名运动员判定（共 12 次调用，约 1–5 分钟）")
    args = ["--config", str(HERE / "judges_deepseek.json"), "--target", f"main={TARGET}",
            "--negative_control", str(nc), "--out", str(OUT), "--workers", "3"]
    run_judges.main(args + (["--mock"] if MOCK else []))

    zip_path = shutil.make_archive(str(HERE / f"{OUT.name}_结果"), "zip", HERE, OUT.name)
    print("\n========== 完成 ==========")
    print(f"报告：{OUT / 'judge_report.md'}")
    print(f"请把这个文件发回：{zip_path}")


if __name__ == "__main__":
    main()
