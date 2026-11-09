"""用多家大模型生成纠正建议，并逐份做规则检查（run_feedback_api.py 的多模型版本）。

论文正文的结果来自 Claude（run_feedback_api.py 或本脚本的 anthropic 配置，二者请求内容相同）；
其他模型的结果可作为跨模型稳健性比较放在补充材料里。

用法：
    cp generators.example.json generators.json     # 核对模型 ID，按需删减
    python run_generate.py --config generators.json --runs 5 --out gen_results
    python run_generate.py --mock --runs 1 --out gen_mock         # 离线自检

输出（--out/<模型名>/）：
    outputs_run1.json ...   每次生成的结果（格式与 check_faithfulness.py 要求一致，可直接交给 run_judges.py）
    raw_run1_N3.txt ...     模型原始回复
    check_run1.txt ...      规则检查报告
另有 --out/run_log.json（每个模型的请求 / 返回模型名、温度、日期、违规数）和 --out/summary.csv。
"""
import argparse
import contextlib
import csv
import datetime as dt
import io
import json
from pathlib import Path

import check_faithfulness as CF
from llm_clients import LLMError, load_model_configs, make_client, parse_json_reply

HERE = Path(__file__).resolve().parent


def mock_generator(system, user, cfg):
    """离线自检：直接返回示例输出中对应运动员的结果。"""
    case = json.loads(user.split("\n", 1)[1])
    ex = json.loads((HERE / "outputs_example_chat.json").read_text(encoding="utf-8"))
    return json.dumps(ex[case["player"]], ensure_ascii=False)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(HERE / "generators.json"))
    ap.add_argument("--inputs", default=str(HERE / "inputs.json"))
    ap.add_argument("--prompt", default=str(HERE / "system_prompt.md"))
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--only", default=None)
    ap.add_argument("--out", default=str(HERE / "gen_results"))
    ap.add_argument("--mock", action="store_true")
    a = ap.parse_args(argv)

    cfgs = ([dict(name="mock-generator", provider="mock", model="mock", temperature=0, responder=mock_generator)]
            if a.mock else load_model_configs(a.config, a.only))
    system = Path(a.prompt).read_text(encoding="utf-8")
    inp = json.loads(Path(a.inputs).read_text(encoding="utf-8"))
    root = Path(a.out)
    root.mkdir(parents=True, exist_ok=True)
    log = dict(mock=a.mock, started_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), models=[])
    summary = []

    for c in cfgs:
        client = make_client(c)
        d = root / c["name"]
        d.mkdir(exist_ok=True)
        rec = dict(name=c["name"], provider=c["provider"], requested_model=c["model"],
                   temperature=c.get("temperature"), max_tokens=c.get("max_tokens", 4000), returned_models=set(),
                   per_run=[])
        for k in range(1, a.runs + 1):
            outputs, parse_fail, errors = {}, [], []
            for pid, case in inp.items():
                user = "诊断结果：\n" + json.dumps(case, ensure_ascii=False, indent=1)
                try:
                    r = client.chat(system, user)
                except LLMError as e:
                    errors.append(f"{pid}: {e}")
                    outputs[pid] = {"player": pid, "items": [], "closing": "<<调用失败>>"}
                    continue
                rec["returned_models"].add(r.model)
                (d / f"raw_run{k}_{pid}.txt").write_text(r.text, encoding="utf-8")
                try:
                    outputs[pid] = parse_json_reply(r.text)
                    assert isinstance(outputs[pid].get("items"), list)
                except (json.JSONDecodeError, ValueError, AssertionError, AttributeError):
                    parse_fail.append(pid)
                    outputs[pid] = {"player": pid, "items": [], "closing": "<<JSON 解析失败>>"}
            f_out = d / f"outputs_run{k}.json"
            f_out.write_text(json.dumps(outputs, ensure_ascii=False, indent=1), encoding="utf-8")
            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf):
                    n_viol = CF.main([str(a.inputs), str(f_out)])
            except (KeyError, TypeError) as e:              # 输出结构不完整时规则检查本身会出错
                n_viol = None
                buf.write(f"规则检查无法运行（输出结构不完整）：{e!r}\n")
            (d / f"check_run{k}.txt").write_text(buf.getvalue(), encoding="utf-8")
            rec["per_run"].append(dict(run=k, violations=n_viol, json_parse_failures=parse_fail, api_errors=errors))
            summary.append(dict(model=c["name"], run=k, violations=n_viol, parse_failures=len(parse_fail),
                                api_errors=len(errors), file=str(f_out)))
            print(f"{c['name']} 第 {k} 次：违规 {n_viol}，解析失败 {parse_fail or '无'}，调用失败 {len(errors)}")
        rec["returned_models"] = sorted(rec["returned_models"])
        log["models"].append(rec)

    log["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    (root / "run_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
    with open(root / "summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    print("完成，记录见", root / "run_log.json")


if __name__ == "__main__":
    main()
