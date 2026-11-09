# Language-model feedback and multi-model judging

The diagnosis decides *what* is wrong and by how much; a language model only rewrites the issue list as coaching language. This folder contains everything needed to generate the correction cues (Sections 2.10 and 3.5 of the paper) and to check them in two independent ways.

| File | Purpose |
|---|---|
| `system_prompt.md` | System prompt for generating correction cues (Chinese; English translation in Supplementary Material S5). |
| `inputs.json` | Model input: the structured issue lists of N1–N5 and of skilled player S3 (empty list, tests that nothing is invented). |
| `outputs_example_chat.json` | Example output that passes all checks. |
| `run_feedback_api.py` | Generation with Claude through the official Anthropic SDK. The cues reported in the paper (`outputs_example_chat.json`) were generated once in the Claude web app with default settings; this script sends the same prompt and inputs through the API. |
| `run_generate.py` | Same generation for any model (Claude, OpenAI, Gemini, DeepSeek, Qwen, …) for a cross-model comparison. |
| `check_faithfulness.py` | **Rule-based check** C1–C6: one-to-one items, restated degrees/direction/consistency, no new numbers, no new joint-angle channels, no out-of-scope words, correct labels. |
| `judge_prompt.md` | Rubric for the **language-model judges** (J1–J6 per issue, G1–G2 per player). |
| `run_judges.py` | Sends every output to several judge models from different developers, in parallel, with retries and resumable caching. |
| `aggregate_judges.py` | Majority vote, inter-judge agreement (percent agreement, Fleiss' κ, Gwet's AC1, pairwise Cohen's κ), negative-control detection, comparison with the rule-based check, and a report with ready-to-edit sentences for the paper. |
| `make_negative_control.py` | Seeds 10 known errors into a faithful output: 5 literal (the rule-based check should catch them) and 5 semantic (only judges can catch them: reversed cue, contradicting description, cue moved to another phase, inferred cause, unsafe drill). |
| `run_deepseek.py`, `judges_deepseek.json` | One-command judging of `outputs_example_chat.json` and the negative control by DeepSeek-V4-Pro (the run reported in the paper). |
| `judge_deepseek/`, `negative_control/` | **Results reported in the paper**: raw DeepSeek replies with reasons, parsed judgements, run log (model, date, prompt hashes), summary tables and `judge_report.md`; the seeded output and the list of seeds. |
| `llm_clients.py` | One small client for all providers, standard library only; API keys are read from environment variables and never logged. |
| `judges.example.json`, `generators.example.json` | Model lists. Copy to `judges.json` / `generators.json`, check the model IDs in each provider's console and delete the ones you do not use. |

## Why two checks
The rule-based check is deterministic and transparent but only sees the literal text: it cannot tell whether a cue asks the player to move *against* the deviation or *further into* it. The judges evaluate meaning (J1–J6), and using several models from different developers limits the bias of any single model, including a model's tendency to approve its own outputs. The negative control measures what each check can actually detect. Neither check judges whether the cues are good coaching; that still needs coaches.

## Rubric
Per issue — J1 restatement faithful; J2 description of the movement in the same direction as the deviation; J3 cue asks for a change *opposite* to the deviation; J4 cue and drill stay within that joint and phase; J5 drill is feasible without a ball or at slow speed and safe; J6 no inferred cause, ball speed, power, injury, talent or medical statement.
Per player — G1 items correspond one-to-one with the list (nothing invented for an empty list); G2 labels and closing sentence follow the rules.

## Running

```bash
cd llm_feedback
cp judges.example.json judges.json          # edit model IDs; add "disabled": true to skip a model
cp .env.example .env                         # fill in the keys you have
set -a; source .env; set +a

# 1. generate the cues (final run for the paper; 5 repeats at temperature 0)
python run_feedback_api.py --model claude-opus-5-5 --temperature 0 --runs 5
#    optional cross-model comparison:
python run_generate.py --config generators.json --runs 5 --out gen_results

# 2. negative control seeded into one faithful output
python make_negative_control.py inputs.json results/outputs_run1.json --out negative_control

# 3. judging (each judge sees the issue list and the output for one player at a time)
python run_judges.py --config judges.json --target main=results/outputs_run1.json \
       --negative_control negative_control --out judge_results
#    add more targets to judge several runs or models, e.g. --target gpt=gen_results/gpt/outputs_run1.json
#    rerunning the same command only repeats failed or missing calls; --force reruns everything
```

Read `judge_results/judge_report.md` first. Every unit failed by any judge is listed in `flagged_for_review.csv` with the judges' reasons: judges can be wrong, so decide for each whether it is a real problem (revise the prompt and regenerate) or a false alarm (report it as such).

Offline check of the whole pipeline (no network, no keys; the numbers are simulated and must not be reported):

```bash
python run_judges.py --mock --target main=outputs_example_chat.json --negative_control negative_control --out judge_mock
python -m unittest discover -s ../tests
```

## What to report
`run_log.json` records, for every model, the provider, the requested and the **returned** model name, temperature, number of calls, token counts, start and end time, and the SHA-256 of the prompt, inputs and judged outputs. Report the returned model names and dates in Section 2.10 and put `judge_prompt.md` and `judge_report.md` in Supplementary Material S5.

## Cost
One judging pass is (number of judges) × (number of judged outputs) × 6 players calls; with five judges, the final output and the negative control that is 60 short calls.
