# PyTorch Foundations

## Why this module exists

I built this small module to practice the full loop of a PyTorch classification project without pretending that a toy dataset is a real benchmark. My goal was to make one idea unmistakable: a model can look excellent when the data split quietly leaks session-specific clues from training into testing.

## What it demonstrates

This repository compares two ways to split synthetic turtle observations:

- An image-level split that treats every image independently.
- A session-aware split that keeps all images from the same capture session together.

The synthetic observations are designed so that images from one session share stronger style cues than the small image-to-image noise. That makes it easy to see how random splitting can reward memorizing the session instead of recognizing the turtle.

## Experiment

The experiment generates 256 synthetic samples with 24 features each, spread across 8 turtle identities and 32 capture sessions. I train the same small neural network twice:

- Once after an image-level split.
- Once after a session-aware split.

Both runs use the same seed and the same training settings so the split strategy is the main difference.

## Quick start

```bash
make setup
make train
make check
```

The training command writes a JSON report to `artifacts/metrics.json`.

## Verified result

The verified `make train` run used `--epochs 40 --seed 42` and produced this dataset summary:

- Samples: 256
- Feature dimension: 24
- Turtle classes: 8
- Capture sessions: 32

The image-level split reported:

- Overlapping sessions across train and test: 28
- Test accuracy: 1.000
- Test macro F1: 1.000
- Final train loss shown by the CLI: 0.000
- Exact final train loss in the JSON report: `1.0741246114776004e-07`

The session-aware split reported:

- Overlapping sessions across train and test: 0
- Test accuracy: 0.875
- Test macro precision: 0.8125
- Test macro recall: 0.8750
- Test macro F1: 0.8333
- Final train loss shown by the CLI: 0.000
- Exact final train loss in the JSON report: `5.898374055846034e-08`

The JSON report also includes an 8x8 confusion matrix for each split. Macro-averaged
precision, recall, and F1 prevent the overall accuracy from hiding a class that is
harder to recognize.

For me, that gap is the point of the exercise. The image-level split looks perfect because images from the same capture session can appear in both partitions. The session-aware split blocks that path and gives a more honest educational result.

## What I learned

I learned that data splitting is part of model design, not an afterthought. When grouped examples share background, lighting, camera, and pose cues, a random split can make a classifier look smarter than it really is. I also practiced keeping experiments reproducible with fixed seeds, a small CLI, and automated checks.

## Limitations

- The data is synthetic, so this is an educational example rather than a real conservation benchmark.
- The feature vectors are already numeric observations, not raw images.
- The model is intentionally compact and does not explore regularization, early stopping, or hyperparameter tuning.
- The evaluation uses one deterministic train/test split per strategy rather than repeated grouped cross-validation.

## Repository structure

- `src/pytorch_foundations/data.py`: synthetic observation generation and safe split helpers
- `src/pytorch_foundations/tensor_basics.py`: tensor inspection and feature normalization exercise
- `src/pytorch_foundations/batches.py`: deterministic `DataLoader` mini-batch helper
- `src/pytorch_foundations/metrics.py`: macro classification metrics and confusion matrices
- `src/pytorch_foundations/model.py`: compact classifier
- `src/pytorch_foundations/training.py`: deterministic training and evaluation loop
- `src/pytorch_foundations/cli.py`: reproducible experiment command and JSON report writer
- `tests/`: focused checks for data, model, training, and CLI behavior
- `docs/learning-notes.md`: my plain-language study notes for this module
- `docs/learning-checklist.md`: active-recall checklist for each study session

## Next steps

- Add validation monitoring and early stopping to the training loop.
- Compare repeated grouped splits instead of relying on one deterministic split.
- Replace the toy grouped data with a real, well-documented dataset collected with consent.
