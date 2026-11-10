# Write-up: what worked, what did not, and what the winners did differently

## What moved the score

| Step | Gain |
|---|---|
| Report labels from Claude Opus under the hosts' expert criteria (vs open-model labels) | 0.937 → 0.945 public, same model |
| Self-training students (50% label + 50% out-of-fold teacher) | 0.945 → 0.948 |
| Blending with the public block | 0.948 → 0.958 |
| Test-time crop augmentation on our part | hidden decimals only (rank 315 → 242 public) |
| S6 retrained on our labels, then as a teacher for fastMRI | single model 0.950 → 0.955 public |
| S6 + fastMRI student inside the final blend | 0.958 → **0.959** public, 0.946 private |

## What did not work (measured)

- Training on AMD MI210 / ROCm: about −0.010 vs CUDA for the same recipe. We moved all training to CUDA.
- A bigger backbone (ConvNeXt-Small) inside our multiple-instance model: tie with Tiny.
- Self-supervised pretraining (LeJEPA, LocaLeJEPA) before fine-tuning: worse than an ImageNet start by 0.004–0.008.
- Down-weighting report-silent cells, propensity reweighting, criterion-state labels: no gain or worse on gold.
- Public fused label sets as extra targets: tie on every independent ruler.
- Larger or smaller crops (352 / 224 px), alternate series at test time, a metadata stacker.

## How the top solutions differed

Our text-only labels scored 0.92 on the 58 expert-labelled studies, at or above every other team's text-only labels we saw (0.89–0.91). The gap to the top came from elsewhere:

1. **LLM agents that read the images to make labels**, not just the report. The top teams ran multi-turn agents (often through Codex / Claude Code in non-interactive mode) that looked at slices and checked each finding against the criteria. Reported gains:
   - 2nd: 0.899 → 0.942 on gold;
   - 10th: +0.017 private.
2. **Telling the model where to look:** anatomy boxes, tiles or zoomed crops. We had this only inside S6.
3. **Many rounds of self-training** (5–10, with calibrated teachers), where we ran one.
4. **Scanner-grouped validation folds.**
5. **Trusting their own models over public ones.** 16th place, with a recipe close to ours, gave its own models 90% of the blend.

## Lessons for the next competition

1. **Build labels from what the graders actually judge.** If experts grade images, read the images.
2. **Model the label source's bias** (sites, scanners, report styles) during training, and validate within groups.
3. **Measure the noise of each check** (gold set, folds, public split) and fix the acceptance rule before running.
4. **New information** (labels, data, views) **beats tuning** seeds, backbones or blend weights.
5. **Time the inference platform early, and chain GPU jobs** so the machine never sits idle.
