"""汇总多模型评审结果：多数票、评审间一致性、阴性对照检出率，并与规则检查对比。

一般由 run_judges.py 自动调用；也可以单独重算：
    python aggregate_judges.py --out judge_results --target main=results/outputs_run1.json --negative_control negative_control

输出（--out 目录）：
    judgments_long.csv            每个评审 × 目标 × 运动员 × 条目 × 标准 一行（含理由）
    panel_decisions.csv           每个单元的多数票结果、判不通过的评审名单
    per_judge_summary.csv         每个评审在每份目标上的不通过数（按标准）
    agreement.csv                 评审间一致性：完全一致比例、Fleiss κ、Gwet AC1，两两 Cohen κ
    negative_control_detection.csv  每个植入错误是否被各评审 / 多数票 / 规则检查发现
    flagged_for_review.csv        正式输出中被任一评审判为不通过的单元及理由（需作者人工核对）
    judge_report.md               以上内容的可读版本，末尾附可直接改写进论文的英文句子
"""
import argparse
import csv
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

import check_faithfulness as CF

ITEM_CRITERIA = ["J1", "J2", "J3", "J4", "J5", "J6"]
GLOBAL_CRITERIA = ["G1", "G2"]
CRIT_EN = {"J1": "restatement faithful", "J2": "description consistent", "J3": "correction direction",
           "J4": "scope (joint/phase)", "J5": "drill feasible/safe", "J6": "no out-of-scope inference",
           "G1": "one-to-one items", "G2": "labels and closing"}


# ---------------------------------------------------------------- 一致性统计
def fleiss_kappa(rows):
    """rows: 每个单元一行 [n_fail, n_pass]，各行评审人数相同。"""
    rows = [r for r in rows if sum(r) > 1]
    if not rows:
        return float("nan")
    n = sum(rows[0])
    N = len(rows)
    p_j = [sum(r[j] for r in rows) / (N * n) for j in range(2)]
    P_i = [(sum(x * x for x in r) - n) / (n * (n - 1)) for r in rows]
    Pbar, Pe = sum(P_i) / N, sum(p * p for p in p_j)
    return float("nan") if Pe >= 1 else (Pbar - Pe) / (1 - Pe)


def gwet_ac1(rows):
    """二分类 Gwet AC1；在"几乎全部通过"的高流行率数据上比 κ 稳定。"""
    rows = [r for r in rows if sum(r) > 1]
    if not rows:
        return float("nan")
    n = sum(rows[0])
    N = len(rows)
    Pa = sum((sum(x * x for x in r) - n) / (n * (n - 1)) for r in rows) / N
    pi = sum(r[0] for r in rows) / (N * n)
    Pe = 2 * pi * (1 - pi)
    return float("nan") if Pe >= 1 else (Pa - Pe) / (1 - Pe)


def cohen_kappa(a, b):
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return float("nan") if pe >= 1 else (po - pe) / (1 - pe)


def fmt(x, nd=2):
    return "—" if x is None or x != x else f"{x:.{nd}f}"


# ---------------------------------------------------------------- 读取
def load_long(out, judges, targets, inp):
    rows = []
    for j in judges:
        for t in targets:
            d = out / "parsed" / j / t
            if not d.exists():
                continue
            for f in sorted(d.glob("r*_*.json")):
                rep, pid = f.stem.split("_", 1)
                rec = json.loads(f.read_text(encoding="utf-8"))
                if pid not in inp:
                    continue
                jd = rec.get("judgement") if rec.get("valid") else None
                ranks = [a["rank"] for a in inp[pid]["issues"]]
                if jd is None:
                    for r in ranks:
                        for c in ITEM_CRITERIA:
                            rows.append(dict(judge=j, target=t, rep=int(rep[1:]), player=pid, rank=r, criterion=c,
                                             passed=None, reason=f"[无效输出] {rec.get('error', '')}"))
                    for c in GLOBAL_CRITERIA:
                        rows.append(dict(judge=j, target=t, rep=int(rep[1:]), player=pid, rank=0, criterion=c,
                                         passed=None, reason=f"[无效输出] {rec.get('error', '')}"))
                    continue
                for it in jd["items"]:
                    for c in ITEM_CRITERIA:
                        rows.append(dict(judge=j, target=t, rep=int(rep[1:]), player=pid, rank=it["rank"],
                                         criterion=c, passed=it[c]["pass"], reason=it[c].get("reason", "")))
                for c in GLOBAL_CRITERIA:
                    g = jd["global"][c]
                    rows.append(dict(judge=j, target=t, rep=int(rep[1:]), player=pid, rank=0, criterion=c,
                                     passed=g["pass"], reason=g.get("reason", "")))
    return rows


def per_judge_votes(rows):
    """把同一评审的多次重复合并为一票（多数；平票按不通过，偏保守）。返回 {(t,p,r,c): {judge: passed}} 和重复一致率。"""
    by = defaultdict(lambda: defaultdict(list))
    reasons = defaultdict(dict)
    for x in rows:
        k = (x["target"], x["player"], x["rank"], x["criterion"])
        by[k][x["judge"]].append(x["passed"])
        if x["passed"] is False:
            reasons[k].setdefault(x["judge"], x["reason"])
    votes, stab = {}, defaultdict(lambda: [0, 0])
    for k, jd in by.items():
        votes[k] = {}
        for j, vs in jd.items():
            vs = [v for v in vs if v is not None]
            if not vs:
                votes[k][j] = None
                continue
            n_fail = sum(not v for v in vs)
            votes[k][j] = not (n_fail >= len(vs) - n_fail)
            if len(vs) > 1:
                stab[j][0] += int(len(set(vs)) == 1)
                stab[j][1] += 1
    return votes, reasons, {j: a / b for j, (a, b) in stab.items() if b}


def write_csv(path, rows, cols=None):
    rows = list(rows)
    cols = cols or (list(rows[0].keys()) if rows else [])
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------- 主函数
def run(out, inputs, targets, seeds=None, judges=None, mock=False):
    out = Path(out)
    inp = json.loads(Path(inputs).read_text(encoding="utf-8"))
    judges = judges or sorted(p.name for p in (out / "parsed").iterdir() if p.is_dir())
    rows = load_long(out, judges, list(targets), inp)
    if not rows:
        print("没有可汇总的评审结果。")
        return
    write_csv(out / "judgments_long.csv", rows)
    votes, reasons, stability = per_judge_votes(rows)

    # 多数票
    panel = []
    for k in sorted(votes, key=lambda k: (k[0], list(inp).index(k[1]), k[2], k[3])):
        v = votes[k]
        valid = {j: x for j, x in v.items() if x is not None}
        fails = sorted(j for j, x in valid.items() if x is False)
        panel.append(dict(target=k[0], player=k[1], rank=k[2], criterion=k[3], n_judges=len(valid),
                          n_fail=len(fails), panel_fail=len(fails) * 2 > len(valid) if valid else None,
                          any_fail=bool(fails), failed_by=";".join(fails),
                          reasons=" | ".join(f"{j}: {reasons[k].get(j, '')}" for j in fails)))
    write_csv(out / "panel_decisions.csv", panel)
    pmap = {(p["target"], p["player"], p["rank"], p["criterion"]): p for p in panel}

    # 每个评审的不通过数
    summ = []
    for t in targets:
        for j in judges:
            ks = [k for k in votes if k[0] == t and j in votes[k]]
            if not ks:
                continue
            row = dict(target=t, judge=j, units=len(ks), invalid=sum(votes[k][j] is None for k in ks),
                       fails=sum(votes[k][j] is False for k in ks),
                       repeat_stability=fmt(stability.get(j)) if j in stability else "")
            for c in ITEM_CRITERIA + GLOBAL_CRITERIA:
                row[c] = sum(votes[k][j] is False for k in ks if k[3] == c)
            summ.append(row)
    write_csv(out / "per_judge_summary.csv", summ)

    # 评审间一致性（只用所有评审都给出有效判断的单元）
    agree = []
    for scope in list(targets) + ["ALL"]:
        ks = [k for k in votes if (scope == "ALL" or k[0] == scope)
              and all(votes[k].get(j) is not None for j in judges)]
        if len(judges) < 2 or not ks:
            continue
        tab = [[sum(votes[k][j] is False for j in judges), sum(votes[k][j] is True for j in judges)] for k in ks]
        agree.append(dict(scope=scope, pair="ALL", units=len(ks),
                          all_agree=sum(min(r) == 0 for r in tab) / len(tab),
                          fleiss_kappa=fleiss_kappa(tab), gwet_ac1=gwet_ac1(tab),
                          fail_rate=sum(r[0] for r in tab) / (len(tab) * len(judges))))
        for a, b in itertools.combinations(judges, 2):
            va = [votes[k][a] is False for k in ks]
            vb = [votes[k][b] is False for k in ks]
            agree.append(dict(scope=scope, pair=f"{a} vs {b}", units=len(ks),
                              all_agree=sum(x == y for x, y in zip(va, vb)) / len(ks),
                              fleiss_kappa=cohen_kappa(va, vb), gwet_ac1=float("nan"),
                              fail_rate=(sum(va) + sum(vb)) / (2 * len(ks))))
    write_csv(out / "agreement.csv", agree, ["scope", "pair", "units", "all_agree", "fleiss_kappa", "gwet_ac1",
                                             "fail_rate"])

    # 规则检查
    rule = {}
    for t, p in targets.items():
        rule[t] = CF.check(inp, json.loads(Path(p).read_text(encoding="utf-8")))

    # 阴性对照
    det_rows, seed_locs = [], set()
    if seeds and "seeded" in targets:
        src = json.loads(Path(seeds["source"]).read_text(encoding="utf-8")) if Path(seeds["source"]).exists() else None
        rule_src = CF.check(inp, src) if src else {pid: [] for pid in inp}
        for s in seeds["seeds"]:
            pid, rk = s["player"], s["rank"]
            locs = [(pid, rk if c.startswith("J") else 0, c) for c in s["expect_judge"]]
            if s["field"] == "items":                         # 删除条目：该条任何一项判不通过也算发现
                locs += [(pid, rk, c) for c in ITEM_CRITERIA]
            seed_locs |= {(pid, rk)} | {(pid, 0)} if s["field"] in ("items", "note") else {(pid, rk)}
            row = dict(seed=s["id"], kind=s["kind"], player=pid, rank=rk, desc=s["desc"],
                       expect_judge="/".join(s["expect_judge"]), expect_rule="/".join(s["expect_rule"]) or "—")
            for j in judges:
                hit = any(votes.get(("seeded", p_, r_, c_), {}).get(j) is False for p_, r_, c_ in locs)
                loose = hit or any(votes.get(("seeded", pid, rk, c_), {}).get(j) is False for c_ in ITEM_CRITERIA)
                row[j] = "✓" if hit else ("~" if loose else "✗")
            row["panel_majority"] = "✓" if any(pmap.get(("seeded",) + loc, {}).get("panel_fail") for loc in locs) else "✗"
            new = [f for f in rule["seeded"][pid] if f not in rule_src.get(pid, [])]
            rhit = any(c in s["expect_rule"] and (r is None or r == rk) for c, r, _ in new)
            row["rule_check"] = "✓" if rhit else "✗"
            det_rows.append(row)
        write_csv(out / "negative_control_detection.csv", det_rows)

    # 正式输出中被判不通过的单元：需人工核对
    flagged = [p for p in panel if p["target"] != "seeded" and p["any_fail"]]
    write_csv(out / "flagged_for_review.csv", flagged,
              ["target", "player", "rank", "criterion", "n_fail", "n_judges", "panel_fail", "failed_by", "reasons"])

    # ------------------------------------------------------------ 报告
    log = json.loads((out / "run_log.json").read_text(encoding="utf-8")) if (out / "run_log.json").exists() else {}
    R = ["# 多模型评审报告（LLM-as-judge）", ""]
    if mock:
        R += ["> **MOCK 运行：评审由离线规则模拟，仅用于检查流程，数字不能写进论文。**", ""]
    R += ["## 评审模型", "", "| 名称 | provider | 请求的模型 | 返回的模型 | 温度 | 调用次数 | 无效输出 | 错误 |",
          "|---|---|---|---|---|---|---|---|"]
    for s in log.get("judges", []):
        R.append(f"| {s['name']} | {s['provider']} | {s['requested_model']} | {', '.join(s['returned_models'])} | "
                 f"{s['temperature']} | {s['calls']} | {len(s['invalid'])} | {len(s['errors'])} |")
    R += ["", f"评审提示词 SHA-256：`{log.get('judge_prompt_sha256', '')}`；开始 {log.get('started_utc', '')}，"
          f"结束 {log.get('finished_utc', '')}（UTC）", ""]

    for t in targets:
        n_items = sum(len(inp[p]["issues"]) for p in inp)
        R += [f"## 目标：{t}（{targets[t]}）", "",
              f"规则检查违规数：{sum(len(v) for v in rule[t].values())}；条目 {n_items} 条 × 6 项 + 运动员 {len(inp)} 名 × 2 项", "",
              "| 评审 | 无效 | 不通过合计 | " + " | ".join(ITEM_CRITERIA + GLOBAL_CRITERIA) + " |",
              "|---|---|---|" + "---|" * 8]
        for r in summ:
            if r["target"] == t:
                R.append(f"| {r['judge']} | {r['invalid']} | {r['fails']} | " +
                         " | ".join(str(r[c]) for c in ITEM_CRITERIA + GLOBAL_CRITERIA) + " |")
        pf = [p for p in panel if p["target"] == t and p["panel_fail"]]
        af = [p for p in panel if p["target"] == t and p["any_fail"]]
        R += ["", f"多数票判为不通过的单元：{len(pf)}；至少一名评审判为不通过：{len(af)}", ""]
        for a in agree:
            if a["scope"] == t and a["pair"] == "ALL":
                R.append(f"评审间一致性：完全一致 {fmt(a['all_agree'] * 100, 1)}%，Fleiss κ = {fmt(a['fleiss_kappa'])}，"
                         f"Gwet AC1 = {fmt(a['gwet_ac1'])}（{a['units']} 个单元；不通过率很低时 κ 会偏低或无定义，以 AC1 和完全一致比例为准）")
        R.append("")

    if det_rows:
        R += ["## 阴性对照：植入错误的检出", "",
              "✓ = 在预期标准上判不通过；~ = 只在该条的其他标准上判不通过；✗ = 未发现", "",
              "| 错误 | 类型 | 位置 | 内容 | 预期评审标准 | " + " | ".join(judges) + " | 多数票 | 规则检查 |",
              "|---|---|---|---|---|" + "---|" * (len(judges) + 2)]
        for r in det_rows:
            R.append(f"| {r['seed']} | {r['kind']} | {r['player']} #{r['rank']} | {r['desc']} | {r['expect_judge']} | " +
                     " | ".join(r[j] for j in judges) + f" | {r['panel_majority']} | {r['rule_check']} |")
        nonseed = [p for p in panel if p["target"] == "seeded" and p["panel_fail"]
                   and (p["player"], p["rank"]) not in seed_locs]
        R += ["", f"阴性对照中，多数票在未植入错误的位置判为不通过的单元：{len(nonseed)}", ""]

    if flagged:
        R += ["## 需人工核对的单元（正式输出中被任一评审判为不通过）", "",
              "评审模型也会误判。请逐条阅读理由：确属问题的，修改提示词后重新生成；属于误判的，在论文中如实报告。", "",
              "| 目标 | 运动员 | 条 | 标准 | 不通过 / 评审数 | 多数票 | 理由 |", "|---|---|---|---|---|---|---|"]
        for p in flagged:
            R.append(f"| {p['target']} | {p['player']} | {p['rank'] or '整体'} | {p['criterion']} | "
                     f"{p['n_fail']}/{p['n_judges']} | {'是' if p['panel_fail'] else '否'} | {p['reasons'][:300]} |")
        R.append("")

    # 论文用英文句子（数字自动填入）
    main_t = next((t for t in targets if t != "seeded"), None)
    jl = log.get("judges", [])
    if main_t and jl:
        pf = sum(1 for p in panel if p["target"] == main_t and p["panel_fail"])
        af = sum(1 for p in panel if p["target"] == main_t and p["any_fail"])
        units = sum(1 for p in panel if p["target"] == main_t)
        a_all = next((a for a in agree if a["scope"] == main_t and a["pair"] == "ALL"), None)
        models = "; ".join(f"{s['provider']} {', '.join(s['returned_models']) or s['requested_model']}" for s in jl)
        R += ["## 可改写进论文的句子（英文，请核对后使用）", "",
              f"*Methods.* In addition to the rule-based check, "
              + (f"{len(jl)} language models from different developers ({models}) independently judged" if len(jl) > 1
                 else f"a language model ({models}) judged") + f" each output against the issue list using a fixed rubric "
              "(Supplementary Material S5): for every issue, whether the restatement was faithful (J1), the description of "
              "the movement matched the direction of the deviation (J2), the cue asked for a change opposite to the "
              "deviation (J3), the cue and drill stayed within the joint and phase (J4), the drill could be practised "
              "without a ball or at slow speed (J5) and no cause, performance or injury statement was added (J6); and for "
              "every player, whether the items corresponded one-to-one with the list (G1) and the labels and closing "
              "sentence followed the rules (G2). " + ("A criterion was judged failed when a majority of the judges failed it. " if len(jl) > 1 else "") +
              "Five semantic errors that the rule-based check cannot detect (a reversed cue, a description contradicting "
              "the deviation, a cue moved to another phase, an inferred cause and an unsafe drill) were seeded together "
              "with five literal errors as a negative control.", ""]
        if det_rows:
            sem = [r for r in det_rows if r["kind"] == "semantic"]
            lit = [r for r in det_rows if r["kind"] == "rule"]
            R += [f"*Results.* Of the {units} judgements on the final outputs, {pf} were failed by a majority of the "
                  f"judges and {af} by at least one judge"
                  + (f"; agreement among the judges was {fmt(a_all['all_agree'] * 100, 1)}% "
                     f"(Gwet's AC1 = {fmt(a_all['gwet_ac1'])})" if a_all else "") +
                  f". In the negative control, the majority vote detected "
                  f"{sum(r['panel_majority'] == '✓' for r in sem)} of {len(sem)} semantic and "
                  f"{sum(r['panel_majority'] == '✓' for r in lit)} of {len(lit)} literal errors, whereas the rule-based "
                  f"check detected {sum(r['rule_check'] == '✓' for r in sem)} and "
                  f"{sum(r['rule_check'] == '✓' for r in lit)}, respectively.", ""]
    (out / "judge_report.md").write_text("\n".join(R), encoding="utf-8")
    print("汇总完成：", out / "judge_report.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--inputs", default=str(Path(__file__).resolve().parent / "inputs.json"))
    ap.add_argument("--target", action="append", default=[])
    ap.add_argument("--negative_control", default=None)
    ap.add_argument("--mock", action="store_true")
    a = ap.parse_args()
    targets = dict(t.split("=", 1) for t in a.target)
    seeds = None
    if a.negative_control:
        targets["seeded"] = str(Path(a.negative_control) / "outputs_seeded.json")
        seeds = json.loads((Path(a.negative_control) / "seeds.json").read_text(encoding="utf-8"))
    run(a.out, a.inputs, targets, seeds, mock=a.mock)


if __name__ == "__main__":
    main()
