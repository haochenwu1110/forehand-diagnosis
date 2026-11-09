"""自动核对大模型反馈是否忠实于诊断结果（规则检查）。

检查项：
  C1 条目一一对应：条数、顺序、关节、阶段与输入一致
  C2 复述完整：方向文字、"约 X°"、出现比例都出现在 restatement 中
  C3 无新增数字：全文数字只能来自输入（度数、比例、相关度数、序号）
  C4 无新增指标：提到的关节角指标只能是该运动员问题清单中的（含 related）
  C5 无越界内容：不出现球速、力量、受伤、医学等词
  C6 标注正确：note 与 agreement / novice_common 一致；closing 符合规则

用法：
    python check_faithfulness.py inputs.json outputs.json            # 打印报告（论文 2.10 / 3.5 节）
    python check_faithfulness.py inputs.json outputs.json --json r.json   # 另存逐条结果，供 aggregate_judges.py 对比

规则检查只能发现"字面上"的不忠实（数字、词表、条数）；纠正方向是否反了、动作描述是否与偏差矛盾等
语义问题由 run_judges.py 的多模型评审补充。
"""
import json
import re
import sys

# 每个通道允许出现的指标词（用于 C4）；不在本人清单里的指标词出现即违规
TOKENS = {
    "pelvis_rot_imu": ["骨盆转动", "骨盆"], "thorax_rot_imu": ["躯干转动", "胸口"], "separation_imu": ["分离角"],
    "trunk_forward_lean": ["躯干前倾", "前倾"], "trunk_lateral_tilt": ["躯干侧倾", "侧倾"],
    "shoulder_elev_R": ["右肩", "右上臂", "右臂"], "uarm_forward_R": ["右上臂", "右臂"], "elbow_flex_R": ["右肘"],
    "shoulder_elev_L": ["左肩", "左臂"], "elbow_flex_L": ["左肘"],
    "hip_flex_L": ["左髋"], "hip_flex_R": ["右髋", "右腿"], "knee_flex_L": ["左膝", "左腿"], "knee_flex_R": ["右膝"],
}
REL_CH = {"躯干转动": "thorax_rot_imu"}
BANNED = ["球速", "力量", "发力", "受伤", "损伤", "伤病", "疼", "医", "康复", "天赋", "水平差"]


def nums(s):
    return re.findall(r"\d+(?:\.\d+)?", s)


def check_player(case, o):
    """返回该运动员的违规列表，每条为 (检查项, 涉及的 rank 或 None, 说明)。"""
    issues = case["issues"]
    fails = []
    # C1
    if len(o["items"]) != len(issues):
        fails.append(("C1", None, f"C1 条数 {len(o['items'])} ≠ {len(issues)}"))
    for a, b in zip(issues, o["items"]):
        if (a["rank"], a["joint"], a["phase"]) != (b["rank"], b["joint"], b["phase"]):
            fails.append(("C1", a["rank"], f"C1 第 {a['rank']} 条关节/阶段不一致"))
    # C2
    for a, b in zip(issues, o["items"]):
        r = b["restatement"]
        for need in (a["direction"], f"约 {a['degrees']}°", a["consistency"]):
            if need not in r:
                fails.append(("C2", a["rank"], f"C2 第 {a['rank']} 条缺少「{need}」"))
    # C3
    allowed = set()
    for a in issues:
        allowed |= {str(a["rank"]), str(a["degrees"]), a["consistency"].rstrip("%")}
        for x in a.get("related", []):
            allowed.add(str(x["degrees"]))
    text_all = json.dumps({k: v for k, v in o.items() if k != "player"}, ensure_ascii=False)
    extra = sorted(set(nums(text_all)) - allowed)
    if extra:
        fails.append(("C3", None, f"C3 出现输入中没有的数字 {extra}"))
    # C4
    chans = {a["channel"] for a in issues} | {REL_CH[x["joint"]] for a in issues for x in a.get("related", [])}
    ok_tokens = {t for c in chans for t in TOKENS[c]}
    body = "".join(it["looks_like"] + it["cue"] + it["drill"] + it["restatement"] for it in o["items"])
    for c, toks in TOKENS.items():
        for t in toks:
            if t in body and t not in ok_tokens:
                fails.append(("C4", None, f"C4 出现清单外的指标词「{t}」"))
    # C5
    for w in BANNED:
        if w in text_all:
            fails.append(("C5", None, f"C5 出现越界词「{w}」"))
    # C6
    for a, b in zip(issues, o["items"]):
        exp = []
        if a["agreement"] == "仅平均模板报出":
            exp.append("确定性较低，建议教练先观察核实")
        if a["novice_common"]:
            exp.append("新手常见问题")
        if b["note"] != "；".join(exp):
            fails.append(("C6", a["rank"], f"C6 第 {a['rank']} 条 note 应为「{'；'.join(exp)}」"))
    exp_close = "未发现需要纠正的问题。" if not issues else "以上建议需由教练结合现场观察确认。"
    if o["closing"] != exp_close:
        fails.append(("C6", None, "C6 closing 不符合规则"))
    return fails


def check(inp, out):
    """{pid: [(检查项, rank, 说明), ...]}，键顺序与 inputs 相同。"""
    return {pid: check_player(case, out[pid]) for pid, case in inp.items()}


def main(argv):
    args = [x for x in argv if not x.startswith("--")]
    f_in = args[0] if len(args) > 0 else "inputs.json"
    f_out = args[1] if len(args) > 1 else "outputs_opus.json"
    inp = json.load(open(f_in, encoding="utf-8"))
    out = json.load(open(f_out, encoding="utf-8"))
    res = check(inp, out)
    n_fail = 0
    for pid, fails in res.items():
        n_fail += len(fails)
        n = len(inp[pid]["issues"])
        print(f"{pid}: {n} 条问题 → {'全部通过' if not fails else '; '.join(f[2] for f in fails)}")
    print("违规总数:", n_fail)
    if "--json" in argv:
        f_json = argv[argv.index("--json") + 1]
        json.dump({pid: [dict(check=c, rank=r, detail=d) for c, r, d in fails] for pid, fails in res.items()},
                  open(f_json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return n_fail


if __name__ == "__main__":
    main(sys.argv[1:])
