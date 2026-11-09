"""在本机把原始 CSV 去标识化后放进公开包（原始文件名含姓名，内容没有表头和姓名）。

用法：python deidentify_raw.py <网球实验文件夹> <输出文件夹>
只复制论文用到的 15 人，文件改名为 S1.csv … N5.csv，内容不改。
"""
import shutil
import sys
from pathlib import Path

MAP = {"A1": "S1", "A2": "S2", "A3": "S3", "A4": "S4", "A5": "S5", "B1": "S6", "B4": "S7", "B5": "S8",
       "C1": "S9", "C3": "S10", "D1": "N1", "D2": "N2", "D3": "N3", "D4": "N4", "D5": "N5"}
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
dst.mkdir(parents=True, exist_ok=True)
done = set()
for f in sorted(src.glob("*.csv")):
    code = f.name.split("-")[0]
    if code in MAP:
        shutil.copyfile(f, dst / f"{MAP[code]}.csv")
        done.add(code)
missing = sorted(set(MAP) - done)
print(f"已复制 {len(done)} 个文件到 {dst}" + (f"；缺少 {missing}" if missing else ""))
