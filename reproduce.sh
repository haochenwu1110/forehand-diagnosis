#!/usr/bin/env bash
# Reproduce all diagnostic results from the public data in data/ (about 2–3 minutes on a laptop CPU).
# Outputs go to out/; compare them with results/. The CCAE comparison (M3_ccae) additionally needs PyTorch.
set -euo pipefail
cd "$(dirname "$0")/code"
export PAPERB_UNIFORM_TIER=1 MPLBACKEND=Agg
OUT=../out
METHODS=M0_mean,M1_knn1,M1_knn3,M1_knn5,M1_knn10,M2_pca80,M2_pca90,M2_pca95,M2_pca98
python -c "import torch" 2>/dev/null && METHODS=$METHODS,M3_ccae,M3_ccae_noopt || echo "PyTorch not found: skipping M3_ccae"
if [ -d ../raw ]; then
  python convert_csv.py --in_dir ../raw --groups groups_public.csv --fps 60 --follow_s 0.35 --stroke_len 61 --out $OUT/prep
  python - <<'PY'
import numpy as np
a = np.load("../out/prep/angles_norm.npz", allow_pickle=True); b = np.load("../data/joint_angles_normalised.npz", allow_pickle=True)
o = [list(a["keys"]).index(k) for k in b["keys"]]
print(f"raw -> joint angles: max difference {np.abs(a['X'][o] - b['X']).max():.1e} deg; "
      f"QC identical: {(a['qc_pass'][o] == b['qc_pass']).all()}")
PY
fi
python localize.py   --prep_dir ../data --out_dir $OUT/group_differences
python layer2_eval.py --prep_dir ../data --out_dir $OUT/reference_models --step1_dir $OUT/group_differences --methods $METHODS
python layer3.py     --prep ../data --step1 $OUT/group_differences --out $OUT/diagnosis
python paper_figures_v2.py --prep ../data --layer2 ../results/reference_models --layer3 $OUT/diagnosis --out $OUT/figures
python make_cards.py $OUT/cards ../data $OUT/diagnosis
cd ..
python - <<'PY'
import json
for n in range(1, 6):
    a = json.load(open(f"out/diagnosis/findings_N{n}.json", encoding="utf-8"))
    b = json.load(open(f"results/diagnosis/findings_N{n}.json", encoding="utf-8"))
    print(f"N{n}: {'identical to results/' if a == b else 'DIFFERENT from results/'}")
PY
python -m unittest discover -s tests
