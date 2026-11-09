"""多模型评审：用 Claude 和其他大模型的 API 逐条判定纠正建议是否忠实于诊断（补充规则检查 check_faithfulness.py）。

规则检查只能发现字面错误（数字、词表、条数）；评审模型按 judge_prompt.md 的 8 项标准做语义判断：
  逐条 J1 复述忠实 / J2 描述一致 / J3 纠正方向正确 / J4 范围限定 / J5 练习可行 / J6 无越界推断
  整体 G1 条目对应 / G2 标注与结尾
多名不同厂商的评审模型独立打分，取多数票，并报告评审间一致性；用植入错误的阴性对照检验评审本身能否发现错误。

用法：
    export ANTHROPIC_API_KEY=... OPENAI_API_KEY=... GEMINI_API_KEY=... DEEPSEEK_API_KEY=... DASHSCOPE_API_KEY=...
    cp judges.example.json judges.json         # 按需要增删模型、核对模型 ID
    # 1) 先生成（或使用已有的）纠正建议，例如 results/outputs_run1.json
    # 2) 构造阴性对照
    python make_negative_control.py inputs.json results/outputs_run1.json --out negative_control
    # 3) 评审：正式输出 + 阴性对照
    python run_judges.py --config judges.json --target main=results/outputs_run1.json \
        --negative_control negative_control --out judge_results
    # 离线自检（不联网、不花钱，结果不能用于论文）：
    python run_judges.py --mock --target main=outputs_example_chat.json --negative_control negative_control --out judge_mock

输出（--out 目录）：
    raw/<评审>/<目标>/r<k>_<运动员>.txt      模型原始回复
    parsed/<评审>/<目标>/r<k>_<运动员>.json  解析后的判定（已存在则跳过，可断点续跑；--force 重跑）
    run_log.json                             每个评审的 provider、请求 / 返回的模型名、温度、调用次数、token、时间、提示词哈希
    以及 aggregate_judges.py 生成的汇总表和 judge_report.md
"""
import argparse
import datetime as dt
import hashlib
import json
import random
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import aggregate_judges
from llm_clients import LLMError, load_model_configs, make_client, parse_json_reply

HERE = Path(__file__).resolve().parent
ITEM_CRITERIA = ["J1", "J2", "J3", "J4", "J5", "J6"]
GLOBAL_CRITERIA = ["G1", "G2"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def user_message(case, feedback):
    return ("## 诊断（算法输出，作为标准）\n" + json.dumps(case, ensure_ascii=False, indent=1) +
            "\n\n## 待评审的纠正建议（模型输出）\n" + json.dumps(feedback, ensure_ascii=False, indent=1) +
            "\n\n请按评审标准输出 JSON。")


def validate(j, case):
    """检查评审输出的结构；返回错误说明（空字符串 = 合格）。"""
    if not isinstance(j, dict):
        return "顶层不是 JSON 对象"
    ranks = [a["rank"] for a in case["issues"]]
    items = j.get("items")
    if not isinstance(items, list):
        return "缺少 items 列表"
    got = [it.get("rank") for it in items if isinstance(it, dict)]
    if got != ranks:
        return f"items 的 rank 应为 {ranks}，实际为 {got}"
    for it in items:
        for c in ITEM_CRITERIA:
            v = it.get(c)
            if not isinstance(v, dict) or not isinstance(v.get("pass"), bool):
                return f"第 {it.get('rank')} 条缺少 {c}.pass（布尔值）"
    g = j.get("global")
    if not isinstance(g, dict):
        return "缺少 global"
    for c in GLOBAL_CRITERIA:
        if not isinstance(g.get(c), dict) or not isinstance(g[c].get("pass"), bool):
            return f"global 缺少 {c}.pass（布尔值）"
    return ""


# ---------------------------------------------------------------- 离线 mock 评审（只用于检查流程）
def mock_judge(system, user, cfg):
    """按简单字符串规则"评审"，再按 cfg['noise'] 随机翻转一部分判断，模拟评审之间的分歧。"""
    m = re.search(r"## 诊断（算法输出，作为标准）\n(.*?)\n\n## 待评审的纠正建议（模型输出）\n(.*?)\n\n请按评审标准", user, re.S)
    case, fb = json.loads(m.group(1)), json.loads(m.group(2))
    rng = random.Random(f"{cfg['name']}|{case['player']}|{cfg.get('_rep', 1)}")
    noise = float(cfg.get("noise", 0.0))
    by_rank = {it["rank"]: it for it in fb.get("items", [])}
    items = []
    for a in case["issues"]:
        it = by_rank.get(a["rank"])
        if it is None:
            items.append({"rank": a["rank"], **{c: {"pass": False, "reason": "该条缺失"} for c in ITEM_CRITERIA}})
            continue
        fail = {
            "J1": f"约 {a['degrees']}°" not in it["restatement"] or a["direction"] not in it["restatement"],
            "J2": "看不出明显差别" in it["looks_like"],
            "J3": "继续保持" in it["cue"] or "不用刻意调整" in it["cue"],
            "J4": "不用刻意调整" in it["cue"] or "右肘" in it["cue"] and "elbow" not in a["channel"],
            "J5": "全速" in it["drill"],
            "J6": any(w in it["looks_like"] + it["cue"] for w in ("球速", "柔韧性", "协调性")),
        }
        exp = "；".join(x for x, ok in (("确定性较低，建议教练先观察核实", a["agreement"] == "仅平均模板报出"),
                                        ("新手常见问题", a["novice_common"])) if ok)
        row = {"rank": a["rank"]}
        for c in ITEM_CRITERIA:
            f = fail[c] if rng.random() >= noise else not fail[c]
            row[c] = {"pass": not f, "reason": "通过" if not f else f"[mock] {c} 不通过"}
        items.append(row)
        if it["note"] != exp:
            row["_note_bad"] = True
    g1 = len(fb.get("items", [])) == len(case["issues"])
    exp_close = "未发现需要纠正的问题。" if not case["issues"] else "以上建议需由教练结合现场观察确认。"
    g2 = fb.get("closing") == exp_close and not any(r.pop("_note_bad", False) for r in items)
    out = {"player": case["player"], "items": items,
           "global": {"G1": {"pass": g1, "reason": "通过" if g1 else "[mock] 条数不符"},
                      "G2": {"pass": g2, "reason": "通过" if g2 else "[mock] 标注不符"}}}
    out["overall_pass"] = all(r[c]["pass"] for r in items for c in ITEM_CRITERIA) and g1 and g2
    return json.dumps(out, ensure_ascii=False)


MOCK_JUDGES = [dict(name=f"mock-judge-{i}", provider="mock", model=f"mock-{i}", noise=n, temperature=0)
               for i, n in ((1, 0.0), (2, 0.03), (3, 0.06))]


# ---------------------------------------------------------------- 主流程
def judge_one(client, system, case, feedback, f_raw, f_parsed):
    msg = user_message(case, feedback)
    last_err, resp = "", None
    for attempt in range(2):                         # 结构不合格时带着错误说明重问一次
        u = msg if attempt == 0 else msg + f"\n\n（上一次输出不符合格式要求：{last_err}。请只输出符合格式的 JSON。）"
        resp = client.chat(system, u)
        f_raw.parent.mkdir(parents=True, exist_ok=True)
        f_raw.write_text(resp.text if attempt == 0 else f_raw.read_text(encoding="utf-8") +
                         "\n\n=== 重问后的回复 ===\n" + resp.text, encoding="utf-8")
        try:
            j = parse_json_reply(resp.text)
            last_err = validate(j, case)
        except (json.JSONDecodeError, ValueError) as e:
            j, last_err = None, f"JSON 解析失败：{e}"
        if not last_err:
            break
    rec = dict(judgement=j if not last_err else None, valid=not last_err, error=last_err,
               returned_model=resp.model, usage=resp.usage, latency_s=resp.latency_s, attempts=attempt + 1)
    f_parsed.parent.mkdir(parents=True, exist_ok=True)
    f_parsed.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(HERE / "judges.json"))
    ap.add_argument("--inputs", default=str(HERE / "inputs.json"))
    ap.add_argument("--prompt", default=str(HERE / "judge_prompt.md"))
    ap.add_argument("--target", action="append", default=[],
                    help="待评审的输出，格式 名称=路径，可重复；例如 main=results/outputs_run1.json")
    ap.add_argument("--negative_control", default=None, help="make_negative_control.py 的输出目录（自动加入目标 seeded）")
    ap.add_argument("--repeats", type=int, default=1, help="每个评审对每份输出重复评审的次数（检查自身稳定性）")
    ap.add_argument("--only", default=None, help="只运行这些评审（逗号分隔的 name）")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=str(HERE / "judge_results"))
    ap.add_argument("--force", action="store_true", help="忽略已有结果，全部重跑")
    ap.add_argument("--mock", action="store_true", help="用离线 mock 评审测试流程（不联网，结果不能用于论文）")
    a = ap.parse_args(argv)

    targets = {}
    for t in a.target:
        if "=" not in t:
            raise SystemExit(f"--target 格式应为 名称=路径：{t}")
        k, v = t.split("=", 1)
        targets[k] = Path(v)
    seeds = None
    if a.negative_control:
        nc = Path(a.negative_control)
        targets["seeded"] = nc / "outputs_seeded.json"
        seeds = json.loads((nc / "seeds.json").read_text(encoding="utf-8"))
    if not targets:
        raise SystemExit("请至少给一个 --target 或 --negative_control")
    for k, v in targets.items():
        if not v.exists():
            raise SystemExit(f"找不到目标文件 {k}={v}")

    cfgs = MOCK_JUDGES if a.mock else load_model_configs(a.config, a.only)
    if a.mock:
        for c in cfgs:
            c["responder"] = mock_judge
    if len(cfgs) < 2:
        print("提示：只有 1 个评审模型，无法计算评审间一致性，也无法做多数票。", file=sys.stderr)

    system = Path(a.prompt).read_text(encoding="utf-8")
    inp = json.loads(Path(a.inputs).read_text(encoding="utf-8"))
    outs = {k: json.loads(v.read_text(encoding="utf-8")) for k, v in targets.items()}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    log = dict(mock=a.mock, started_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
               judge_prompt_sha256=sha256(a.prompt), inputs_sha256=sha256(a.inputs),
               targets={k: dict(path=str(v), sha256=sha256(v)) for k, v in targets.items()},
               repeats=a.repeats, judges=[])
    lock = threading.Lock()
    jobs = []
    stats = {}
    for c in cfgs:
        stats[c["name"]] = dict(name=c["name"], provider=c["provider"], requested_model=c["model"],
                                temperature=c.get("temperature"), max_tokens=c.get("max_tokens", 4000),
                                base_url=c.get("base_url"), returned_models=set(), calls=0, skipped_existing=0,
                                invalid=[], errors=[], input_tokens=0, output_tokens=0)
        for rep in range(1, a.repeats + 1):
            cc = dict(c, _rep=rep)
            client = make_client(cc)
            for tname, o in outs.items():
                for pid, case in inp.items():
                    if pid not in o:
                        print(f"警告：{tname} 中没有 {pid}，跳过", file=sys.stderr)
                        continue
                    f_raw = out / "raw" / c["name"] / tname / f"r{rep}_{pid}.txt"
                    f_parsed = out / "parsed" / c["name"] / tname / f"r{rep}_{pid}.json"
                    if f_parsed.exists() and not a.force and \
                            json.loads(f_parsed.read_text(encoding="utf-8")).get("valid"):
                        stats[c["name"]]["skipped_existing"] += 1      # 只跳过合格的结果，不合格的重跑
                        continue
                    jobs.append((c["name"], client, case, o[pid], f_raw, f_parsed, tname, pid, rep))

    print(f"评审 {len(cfgs)} 个 × 目标 {len(outs)} 份 × 运动员 {len(inp)} 名 × 重复 {a.repeats} 次；"
          f"需调用 {len(jobs)} 次（已有结果跳过 {sum(s['skipped_existing'] for s in stats.values())} 次）")
    with ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
        fut = {ex.submit(judge_one, j[1], system, j[2], j[3], j[4], j[5]): j for j in jobs}
        for f in as_completed(fut):
            name, _, _, _, _, _, tname, pid, rep = fut[f]
            s = stats[name]
            try:
                rec = f.result()
            except LLMError as e:
                with lock:
                    s["errors"].append(f"{tname}/{pid}/r{rep}: {e}")
                print(f"  ✗ {name} {tname} {pid} r{rep}: {e}", file=sys.stderr)
                continue
            with lock:
                s["calls"] += rec["attempts"]
                s["returned_models"].add(rec["returned_model"])
                u = rec["usage"] or {}
                s["input_tokens"] += int(u.get("input_tokens") or u.get("prompt_tokens") or u.get("promptTokenCount") or 0)
                s["output_tokens"] += int(u.get("output_tokens") or u.get("completion_tokens")
                                          or u.get("candidatesTokenCount") or 0)
                if not rec["valid"]:
                    s["invalid"].append(f"{tname}/{pid}/r{rep}: {rec['error']}")
            flag = "✓" if rec["valid"] else "?"
            ok = rec["judgement"].get("overall_pass") if rec["valid"] else None
            print(f"  {flag} {name:<22} {tname:<8} {pid:<4} r{rep}  overall_pass={ok}")

    for s in stats.values():
        s["returned_models"] = sorted(s["returned_models"])
        log["judges"].append(s)
    log["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    f_log = out / "run_log.json"
    if f_log.exists() and not a.force:              # 断点续跑：保留历次记录
        old = json.loads(f_log.read_text(encoding="utf-8"))
        log["previous_runs"] = old.get("previous_runs", []) + [{k: v for k, v in old.items() if k != "previous_runs"}]
    f_log.write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")

    aggregate_judges.run(out, a.inputs, {k: str(v) for k, v in targets.items()}, seeds,
                         judges=[c["name"] for c in cfgs], mock=a.mock)
    n_err = sum(len(s["errors"]) + len(s["invalid"]) for s in stats.values())
    if n_err:
        print(f"注意：有 {n_err} 次调用失败或输出不合格，详见 {f_log}；重新运行同一命令会只补跑缺失部分。")


if __name__ == "__main__":
    main()
