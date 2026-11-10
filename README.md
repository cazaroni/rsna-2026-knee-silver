# RSNA 2026 Knee Abnormality Detection: silver medal solution

**Private leaderboard rank 159 of 6,425 (top 2.5%), silver medal.** Private score 0.946 (public 0.959), macro ROC AUC over 12 findings.

Team *tecitos* (Kaggle: `yorchjais`, `laurachenoweth`). This repository holds the code behind our final submission, cleaned of machine-specific paths. It does not hold any competition data, labels or model weights (see [What is not included](#what-is-not-included)).

## The task

Each training study is a knee MRI exam (several series, about 4,300 studies) with a **free-text radiology report**, but no image-level labels. We had to predict 12 findings per study: ACL, MCL, medial/lateral meniscus tear, medial/lateral/patellofemoral osteoarthritis, effusion, synovitis, Baker's cyst, bone contusion and fracture. The test labels came from **expert radiologists reading the images**, so the core difficulty was turning report text into labels that match what image readers see.

## Approach

1. **Labels from reports with an LLM** (`labels/relabel.py`).
   - Claude Opus read every report under the hosts' published expert criteria. Per finding, it returned a status (explicit positive / explicit negative / uncertain / not mentioned), a probability and an evidence quote.
   - On the 58 expert-labelled studies these labels reached **0.92 macro AUC**, against 0.87 for our earlier open-model labels.
   - Relabelling alone took the same model from 0.937 to 0.945 on the leaderboard.
2. **Image cache** (`data/`).
   - Each series is oriented and resampled to a 384 px uint8 stack.
   - Right knees are mirrored to left (`laterality.py`, with an image-based side classifier `latclf.py` where DICOM tags are missing).
   - Folds are made in `make_folds.py`.
3. **Multiple-instance model** (`training/train_mil2.py`).
   - A study is a bag of 48 windows of three adjacent slices (320 px crops).
   - A timm backbone (ConvNeXt-Tiny or EfficientNet-B0) embeds each window.
   - A per-finding attention head pools the bag into 12 logits.
   - Training uses soft BCE on soft targets, with EMA weights.
4. **Self-training** (`training/make_student_targets.py`, `make_gen2_targets.py`).
   - Students train on 50% LLM label + 50% out-of-fold teacher prediction.
   - Synovitis is centred per scanner using a table fitted on training predictions (`build_site_table.py`), and blended 75/25 with the effusion rank.
5. **External images: fastMRI** (`external_fastmri/`).
   - About 9,400 NYU fastMRI knee studies were converted with the same cache builder and sided.
   - They were pseudo-labelled by our students, then the students were retrained on our data + fastMRI.
   - fastMRI requires registration and a data-use agreement. None of its data or derived files are distributed here.
6. **Anatomy-guided S6 model** (`s6/`).
   - We retrained the public **S6** architecture by heliosli, using nartaa's all-data training code (Apache-2.0, `nartaa/rsna-knee-s6fall-weights`), from scratch on our labels.
   - S6 is a ConvNeXt-Small that reads the full field of view plus 100 mm knee-centred crops placed by an SKM-TEA anatomy network.
   - Then three S6 fold teachers pseudo-labelled fastMRI, and a final S6 student trained on 13,658 studies.

   | Model | Public | Private |
   |---|---|---|
   | public S6 | 0.950 | — |
   | S6 retrained on our labels | 0.951 | 0.940 |
   | **S6 + fastMRI student** | **0.955** | **0.944** |
7. **Final blend** (`inference/`).
   - A Kaggle notebook (2× T4, 9 h limit) runs our models, then the public "block" (Raptor CoAtNet + heliosli R6), and blends per finding by rank: **60% block + 40% S6 + fastMRI**.
   - The notebook builder and patches are in `inference/`. `test_nb.sh` checks a pushed notebook against a local reference before any submission.

## Results

| Submission | Public | Private |
|---|---|---|
| Own-data students only | 0.948 | 0.938 |
| Public block + our students | 0.958 | 0.944 |
| Our models only (S6 + fastMRI + students, no public block) | 0.954 | 0.944 |
| **Final: block 60% + S6 + fastMRI 40%** | **0.959** | **0.946** |

## Repository layout

| Folder | Contents |
|---|---|
| `labels/` | LLM report labelling (Anthropic Message Batches) |
| `data/` | DICOM → cache builder, laterality, folds |
| `training/` | MIL trainer and self-training target builders |
| `external_fastmri/` | fastMRI conversion, siding, pseudo-labelling chains |
| `s6/` | S6 retraining, teachers, fastMRI student and gold evaluation |
| `inference/` | Kaggle notebook builder, blend patches, inference package, configs, submission test |
| `evaluation/` | Paired-bootstrap screens and test-time augmentation scoring |
| `experiments/` | LeJEPA / LocaLeJEPA self-supervised pretraining (did not help; see the write-up) |
| `docs/` | Write-up and lessons learned |

**These are research scripts as they ran,** not a packaged library. Paths point to a placeholder `/workspace` or `/cluster/project` layout, and the scripts assume the competition data is downloaded locally.

## What is not included

- **Competition data:** images, reports and labels. The RSNA data licence does not allow redistribution.
- **Our LLM label file:** it is derived from the competition reports. `labels/relabel.py` regenerates it.
- **fastMRI data and anything derived from it,** including the fastMRI-trained weights. Its data-use agreement does not allow redistribution.
- **Third-party code:** nartaa's S6 trainer and heliosli's anatomy model are used from their public Kaggle datasets, not copied here.

## Write-up

See [`docs/WRITEUP.md`](docs/WRITEUP.md) for what worked, what did not, and how our solution compares with the winning ones.

## Licence

MIT, for our code. Third-party components keep their own licences.
