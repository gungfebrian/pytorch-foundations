# PyTorch Foundations Learning Checklist

Use this page as a short active-recall loop. Check an item only after explaining it without opening the source code.

## Core concepts

- [ ] I can explain what a feature is in this module.
- [ ] I can explain what a label is and how it becomes a training target.
- [ ] I can describe the difference between training, evaluation, and prediction.
- [ ] I can explain why a test set must stay untouched until evaluation.
- [ ] I can identify a capture-session leak in a dataset split.

## PyTorch practice

- [ ] I can read a tensor's shape, dtype, device, and number of elements.
- [ ] I can describe what a forward pass returns from this classifier.
- [ ] I can explain why `CrossEntropyLoss` receives logits and integer class labels.
- [ ] I can trace one optimizer step: zero gradients, forward pass, loss, backward pass, update.
- [ ] I can reproduce the same result by reusing the same seed.

## Experiment review

- [ ] I can state how many samples, classes, and sessions the synthetic dataset contains.
- [ ] I can predict which split should have overlapping sessions before running the command.
- [ ] I can explain why the session-aware score is the more honest educational comparison.
- [ ] I can name one limitation of synthetic numeric features.

## Reflection

After a study session, write three short answers:

1. What did I understand well enough to explain from memory?
2. What error or uncertainty appeared while implementing it?
3. What is the smallest next exercise that would close that gap?
