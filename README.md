# Skilled-referenced structured diagnosis of individual tennis forehand technique — data and code

Data and code for the manuscript *"Skilled-Referenced Structured Diagnosis of Individual Forehand Technique in Tennis Using Full-Body Inertial Motion Capture"* (submitted to *Sports*, MDPI).

The method builds its standard from skilled players only: it measures how much skilled players differ from one another at each joint and phase, subtracts this style variation when diagnosing a novice, and calibrates the reporting threshold so that a skilled player assessed as a newcomer receives about one issue. Each reported issue states the joint, phase, direction and size in degrees. A constrained large language model can then rewrite the issue list as correction cues; a rule-based script and a panel of language models from different developers check that the cues stay faithful to the diagnosis.

## Participants and IDs
15 right-handed players: S1–S5 high-level (National First-Class Athletes), S6–S10 proficient (National Second-Class Athletes), N1–N5 novices. All data are de-identified; no names appear anywhere.

## Repository layout

| Folder | Contents |
|---|---|
| `data/` | `joint_angles_normalised.npz`: 14 joint angles × 125 phase-normalised points per stroke (`X`), with `keys`, `subject`, `group`, `qc_pass`, `channels`, `seg_bounds`; 320 strokes pass quality control. `strokes_meta.csv`: events, phase durations and QC result per stroke. |
| `results/` | Outputs reported in the paper: group-level cluster permutation tests (`group_differences/`), reference-model comparison (`reference_models/`), structured issue lists, threshold calibration, ablation and stability (`diagnosis/`). |
| `code/` | Analysis pipeline (see `code/README.md`). |
| `llm_feedback/` | Language-model application example: system prompt, inputs, generation scripts for Claude and other models, the rule-based faithfulness check, and the multi-model judging pipeline (see `llm_feedback/README.md`). |
| `tests/` | Offline tests (no network, no API keys). |
| `tools/` | De-identification of the raw exports (used to prepare the raw files shared on request). |

## Quick start

```bash
python -m pip install -r requirements.txt
./reproduce.sh            # runs the whole diagnosis from data/ and checks it against results/
```

`reproduce.sh` regenerates the group differences, the reference-model comparison, the issue lists of all novices, the figures and the one-page diagnosis cards, and checks that each novice's issue list is identical to the one in `results/`. Without PyTorch the CCAE reference (M3) is skipped; all other numbers reproduce exactly.

Language-model part:

```bash
cd llm_feedback
python run_judges.py --mock --target main=outputs_example_chat.json --out judge_mock   # offline pipeline check
# real run: copy judges.example.json to judges.json, export the API keys, then see llm_feedback/README.md
```

## Raw data
The raw joint-centre trajectories (Perception Neuron 3 Pro, 60 Hz; 21 joints; 322 strokes in consecutive 61-frame windows) are not included and are available from the corresponding author on reasonable request. With them, `code/convert_csv.py` (see `code/README.md`) recomputes `data/joint_angles_normalised.npz` to within 1e-10°, with identical quality-control results; `reproduce.sh` runs this check automatically if a `raw/` folder is present.

## Requirements
Python ≥ 3.10 with NumPy, SciPy, pandas, matplotlib and Pillow. PyTorch is needed only for the CCAE comparison. The language-model scripts use the Python standard library only (`run_feedback_api.py` additionally uses the official `anthropic` SDK).

## Citation
Please cite the article; see `CITATION.cff`.

## Licence
Code: MIT (`LICENSE`). Data, results and language-model inputs/outputs: CC BY 4.0 (`LICENSE-DATA.md`).
