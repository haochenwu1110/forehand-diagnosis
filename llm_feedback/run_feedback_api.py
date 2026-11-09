"""用 Anthropic API 生成纠正建议，并自动核对忠实性（论文 2.10 / 3.5 节）。

用法（WSL 里）：
    pip install anthropic
    export ANTHROPIC_API_KEY=你的key
    python run_feedback_api.py --model <控制台里的模型名> --temperature 0 --runs 5

输出（都在 results/ 下）：
    outputs_run1.json ...   每次生成的结果（格式与 check_faithfulness.py 要求一致）
    raw_run1_N3.txt ...     模型原始回复（解析失败时可以查看）
    check_run1.txt ...      每次的核对报告
    run_log.json            模型名、温度、日期、SDK 版本、每次的违规数，论文 2.10 节照此填写
"""
import argparse
import datetime as dt
import json
import re
import subprocess
import sys
from pathlib import Path

import anthropic

HERE = Path(__file__).resolve().parent


def parse_json(text):
    """模型偶尔会用 ```json 包起来，去掉后再解析。"""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    return json.loads(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="控制台模型列表里的正式模型名")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--max_tokens", type=int, default=4000)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    system = (HERE / "system_prompt.md").read_text(encoding="utf-8")
    inputs = json.loads((HERE / "inputs.json").read_text(encoding="utf-8"))
    out_dir = HERE / a.out
    out_dir.mkdir(exist_ok=True)
    client = anthropic.Anthropic()          # 从环境变量 ANTHROPIC_API_KEY 读取

    log = dict(requested_model=a.model, temperature=a.temperature, max_tokens=a.max_tokens,
               runs=a.runs, sdk_version=anthropic.__version__,
               started_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
               returned_models=set(), per_run=[])

    for k in range(1, a.runs + 1):
        outputs, parse_fail = {}, []
        for pid, case in inputs.items():
            msg = client.messages.create(
                model=a.model, max_tokens=a.max_tokens, temperature=a.temperature, system=system,
                messages=[{"role": "user",
                           "content": "诊断结果：\n" + json.dumps(case, ensure_ascii=False, indent=1)}])
            text = "".join(b.text for b in msg.content if b.type == "text")
            log["returned_models"].add(msg.model)
            (out_dir / f"raw_run{k}_{pid}.txt").write_text(text, encoding="utf-8")
            try:
                outputs[pid] = parse_json(text)
            except json.JSONDecodeError:
                parse_fail.append(pid)
                outputs[pid] = {"player": pid, "items": [], "closing": "<<JSON 解析失败>>"}
        f_out = out_dir / f"outputs_run{k}.json"
        f_out.write_text(json.dumps(outputs, ensure_ascii=False, indent=1), encoding="utf-8")

        rep = subprocess.run([sys.executable, str(HERE / "check_faithfulness.py"), str(HERE / "inputs.json"),
                              str(f_out)], capture_output=True, text=True, encoding="utf-8")
        (out_dir / f"check_run{k}.txt").write_text(rep.stdout + rep.stderr, encoding="utf-8")
        m = re.search(r"违规总数:\s*(\d+)", rep.stdout)
        n_viol = int(m.group(1)) if m else None
        log["per_run"].append(dict(run=k, violations=n_viol, json_parse_failures=parse_fail))
        print(f"第 {k} 次：违规 {n_viol}，解析失败 {parse_fail or '无'}")

    log["returned_models"] = sorted(log["returned_models"])
    log["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    (out_dir / "run_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
    print("完成，记录见", out_dir / "run_log.json")


if __name__ == "__main__":
    main()
