# Analysis code

Comments in the source are in Chinese; this file describes the pipeline in English. Internal participant codes used during analysis (A1–A5, B1, B4, B5, C1, C3, D1–D5) correspond to S1–S5, S6–S10 and N1–N5; every script accepts either coding.

## Pipeline

| Step | Script | Input → output | Paper |
|---|---|---|---|
| 0 | `convert_csv.py` | raw 21-joint CSVs + `groups.csv` → `angles_norm.npz`, `strokes_meta.csv`, `conversion_report.md` | 2.2–2.3 |
| 1 | `localize.py` | joint angles → group-level cluster permutation tests, group intervals | 2.11, 3.1 |
| 2 | `layer2_eval.py` | joint angles → reference-model comparison (false alarms, injection detection, stability) | 2.5–2.7, 2.9, 3.2 |
| 3 | `layer3.py` | joint angles + step 1 → threshold calibration, structured issue lists (`findings_*.json`), ablation, stability | 2.7–2.9, 3.3–3.4 |
| figures | `paper_figures_v2.py`, `fig1_panels.py`, `fig_skeleton.py`, `panel_d.py` | results → Figures 1–5 | |
| cards | `make_cards.py` | issue lists → one-page diagnosis card per player | |

Shared modules: `config.py` (channels, phases, QC thresholds), `kinematics.py` (joint-angle definitions), `analysis_common.py` (data loading), `reference.py` (M0 mean template, M1 k-nearest neighbours, M2 clipped PCA projection), `reference_ccae.py` (M3 constrained convolutional autoencoder, needs PyTorch), `test_ccae.py` (quick CCAE self-check), `dataset.py` (PyTorch dataset helper), `paper_figures.py` (shared figure style).

## Running from the public data

`data/` already contains the output of step 0, so the analysis starts at step 1 (this is what `../reproduce.sh` does):

```bash
export PAPERB_UNIFORM_TIER=1          # all joints come from one system: no reliability down-weighting
python localize.py    --prep_dir ../data --out_dir ../out/group_differences
python layer2_eval.py --prep_dir ../data --out_dir ../out/reference_models --step1_dir ../out/group_differences \
                      --methods M0_mean,M1_knn1,M1_knn3,M1_knn5,M1_knn10,M2_pca80,M2_pca90,M2_pca95,M2_pca98,M3_ccae,M3_ccae_noopt
python layer3.py      --prep ../data --step1 ../out/group_differences --out ../out/diagnosis      # M0_mean + M2_pca90
```

Running from the raw files (available from the corresponding author on request; place them in `../raw/`):

```bash
python convert_csv.py --in_dir ../raw --groups groups_public.csv --fps 60 --follow_s 0.35 --stroke_len 61 --out ../out/prep
```
`groups_public.csv` assigns S1–S5 to `pro` (high-level), S6–S10 to `amateur` (proficient) and N1–N5 to `novice`. The resulting `angles_norm.npz` equals `data/joint_angles_normalised.npz` to within 1e-10° and gives the same 320 strokes passing quality control.

## Reproducibility check
With the public data, `localize.py`, `layer2_eval.py` (all non-CCAE references) and `layer3.py` reproduce the files in `../results/` exactly, including the injection-detection tables (random seeds are fixed per scenario). Subjects are sorted naturally (S2 before S10) so that the seeds match the original analysis.

## Figure assets
`fig_skeleton.py` needs a mannequin drawing and a sensor photograph that are not redistributed (third-party copyright): pass the folder containing `mannequin.png` and `sensor_pn3.png` as the second argument. `panel_d.py` expects the assembled panels (a)–(c) as `Fig1_abc.png` in its working folder.
