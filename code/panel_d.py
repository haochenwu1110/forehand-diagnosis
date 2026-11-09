"""Fig 1 新增 (d)：受约束的语言模型反馈（应用示例），并与 (a)–(c) 拼接。

用法：python panel_d.py <工作目录>
    工作目录里需要已有 Fig1_abc.png（图 1 的 (a)–(c)，由 fig1_panels.py 的小图手工拼版得到）；
    输出 panel_d.png 和 Fig1_overview.png / .tif。字体用 TeX Gyre Termes（texlive-fonts-extra），没有时退回 DejaVu。
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch  # noqa: E402
from PIL import Image  # noqa: E402

TG = "/usr/share/texmf/fonts/opentype/public/tex-gyre/"
for f in ("regular", "bold", "italic", "bolditalic"):
    if Path(TG + f"texgyretermes-{f}.otf").exists():
        font_manager.fontManager.addfont(TG + f"texgyretermes-{f}.otf")
WD = Path(sys.argv[1] if len(sys.argv) > 1 else "fig1edit")
WD.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": ["TeX Gyre Termes", "DejaVu Sans"], "mathtext.fontset": "stix"})

W, H = 518.4, 92.0                      # pt；宽度与原图相同
S = 4320 / 518.4
DPI = 72 * S
TEAL, TEAL_BG, INK, GRAY, ORANGE = "#1a7f6e", "#eaf5f2", "#222222", "#6b6b6b", "#d9622b"

fig = plt.figure(figsize=(W / 72, H / 72), dpi=DPI)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(H, 0)
ax.axis("off")

# 面板外框（与 (b)(c) 同样的虚线圆角框）
ax.add_patch(FancyBboxPatch((3.5, 8.0), W - 7.0, H - 12.0, boxstyle="round,pad=0,rounding_size=6",
                            fc=TEAL_BG, ec=TEAL, lw=1.0, ls=(0, (1.8, 1.2))))
# 从 (c) 的问题清单引下来的箭头
ax.add_patch(FancyArrowPatch((336, -6), (336, 9), arrowstyle="-|>", mutation_scale=7, color=ORANGE, lw=1.0))
ax.text(8.5, 17.5, "(d)", fontsize=7.56, fontweight="bold", color=INK, va="baseline")
ax.text(20.5, 17.5, "Coaching feedback from the issue list", fontsize=7.56, fontweight="bold", color=TEAL,
        va="baseline")
ax.text(158, 17.5, "— application example: a constrained language model (Claude, claude-opus-5-5) rewords the diagnosis, "
        "and a script and DeepSeek-V4-Pro check it", fontsize=5.6, style="italic", color=TEAL, va="baseline")

boxes = [(10, 112, "Input · N3, issue 1"), (124, 222, "Claude · constrained prompt"),
         (234, 386, "Feedback for the player (in Chinese; translated)"), (398, 510, "Faithfulness checks")]
top, bot = 25.0, 84.0
for x0, x1, title in boxes:
    ax.add_patch(FancyBboxPatch((x0, top), x1 - x0, bot - top, boxstyle="round,pad=0,rounding_size=3",
                                fc="white", ec=TEAL, lw=0.8))
    ax.text(x0 + 4, top + 7.5, title, fontsize=5.6, fontweight="bold", color=TEAL, va="baseline")
for (a0, a1, _), (b0, b1, _) in zip(boxes[:-1], boxes[1:]):
    ax.add_patch(FancyArrowPatch((a1 + 1, (top + bot) / 2), (b0 - 1, (top + bot) / 2), arrowstyle="-|>",
                                 mutation_scale=7, color=TEAL, lw=1.0))



def lines(x, y, rows, size=5.0, gap=6.6, color=INK):
    for i, r in enumerate(rows):
        if isinstance(r, tuple):
            ax.text(x, y + i * gap, r[0], fontsize=size, color=INK, fontweight="bold", va="baseline")
            ax.text(x + r[2], y + i * gap, r[1], fontsize=size, color=color, va="baseline")
        else:
            ax.text(x, y + i * gap, r, fontsize=size, color=color, va="baseline")


lines(14, 41, [("joint", "L knee flexion", 25), ("phase", "backswing → forward", 25),
               ("direction", "less flexed, −48°", 25), ("strokes", "100% in the same direction", 25),
               ("flags", "both references · ★ group-level", 25)])
lines(128, 41, ["• one item per diagnosed issue, same order",
                "• no new joints, phases or numbers",
                "• no claims on speed, power or injury",
                "• fixed fields: restatement · looks like ·",
                "   cue · drill · note"])
lines(238, 41, [("Looks like", "the left leg stays almost straight from backswing", 32),
                ("", "to forward swing; the body does not sink.", 32),
                ("Cue", "bend the left knee and sink in the backswing;", 32),
                ("", "keep it bent through the forward swing.", 32),
                ("Drill", "shadow swings: sink on the left knee, then hold.", 32)])
# 规则脚本（C1–C6）与 DeepSeek 评审（J1–J6、G1–G2）
X0 = 402
ax.text(X0, 40.0, "Rule-based script · C1–C6", fontsize=5.0, fontweight="bold", color=TEAL, va="baseline")
ax.text(X0, 46.4, "16 / 16 novice items · 0 violations", fontsize=4.8, color=INK, va="baseline")
ax.text(X0, 52.4, "seeded errors found: 5 / 10", fontsize=4.5, color=GRAY, va="baseline")
ax.plot([X0, 506], [56.2, 56.2], color=TEAL, lw=0.4, alpha=0.6)
ax.text(X0, 63.4, "Judge model · DeepSeek-V4-Pro", fontsize=5.0, fontweight="bold", color=TEAL, va="baseline")
ax.text(X0, 69.8, "J1–J6, G1–G2: 106 / 108 passed", fontsize=4.8, color=INK, va="baseline")
ax.text(X0, 75.8, "seeded errors found: 10 / 10", fontsize=4.5, color=GRAY, va="baseline")

fig.savefig(WD / "panel_d.png", dpi=DPI, facecolor="white")

# 拼接
abc = Image.open(WD / "Fig1_abc.png").convert("RGB")
d = Image.open(WD / "panel_d.png").convert("RGB")
d = d.resize((abc.width, int(round(d.height * abc.width / d.width))), Image.LANCZOS)
canvas = Image.new("RGB", (abc.width, abc.height + d.height - int(5 * S)), "white")
canvas.paste(abc, (0, 0))
canvas.paste(d.crop((0, int(5 * S), d.width, d.height)), (0, abc.height))
# 箭头穿过 (c) 底边：在原图底部补画箭头上半段
from PIL import ImageDraw  # noqa: E402
dr = ImageDraw.Draw(canvas)
x = int(336 * S)
dr.line([(x, int(349.5 * S)), (x, abc.height)], fill=ORANGE, width=int(1.0 * S))
OUT = str(WD / "Fig1_overview")
canvas.save(OUT + ".png", dpi=(600, 600))
canvas.save(OUT + ".tif", dpi=(600, 600), compression="tiff_lzw")
canvas.save(OUT + ".pdf", resolution=600)
print(canvas.size)
