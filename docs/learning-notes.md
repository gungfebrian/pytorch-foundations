# Learning Notes

## What is a feature?

I think of a feature as one measured clue about an example. In this project, each turtle observation is a vector of numbers. Those numbers stand in for the kind of visual patterns a real image pipeline might extract.

## What is a label?

A label is the answer I want the model to learn. Here, the label is the turtle identity. During training, the model sees features and labels together so it can learn which patterns point to which turtle.

## What is training?

Training is repeated practice. The model makes a prediction, measures how wrong it was, and then nudges its weights to do a little better next time. After enough passes through the training data, the model starts mapping features to labels more reliably.

## What is a prediction?

A prediction is the model's best guess about which class an observation belongs to. The classifier produces one score per turtle class, and the highest score becomes the predicted identity.

## Why the split strategy matters

Photographs from the same capture session share background, lighting, camera, and pose cues, so random splitting can reward memorizing the session instead of recognizing the turtle.

That is why the image-level split in this module looks almost perfect while the session-aware split is meaningfully harder. The second split removes the shortcut.

## What the verified run showed me

In the verified run, the image-level split had 28 overlapping sessions between training and testing and reached 1.000 test accuracy. The session-aware split had 0 overlapping sessions and reached 0.875 test accuracy. I read that difference as evidence that the easier split leaks information the model should not get for free.

I also record macro precision, recall, F1, and a confusion matrix. Macro F1 gives every turtle identity equal weight, which makes the session-aware result easier to inspect than accuracy alone.

Before training, I can fit a scaler on the training partition only and apply those statistics to the test partition. Fitting on all rows would leak information from evaluation into preprocessing, even if the model never sees the test labels.

The training helper can also monitor a session-disjoint validation partition and restore the best checkpoint when validation accuracy stops improving. The returned `best_epoch` and `epochs_ran` make that decision visible in the report.

## Limitations

- I am using synthetic numeric observations instead of a real image dataset.
- One deterministic split per strategy is easy to reproduce, but it is not the same as a full evaluation study.
- The current model is intentionally small and leaves out regularization, early stopping, and repeated grouped evaluation.
- Perfect or near-perfect results here should not be treated as ecological performance claims.

## Active recall

If I had to explain the core lesson from memory, I would say this: when photos from the same capture session appear on both sides of a random split, the model can win by remembering the session's shared cues instead of learning the turtle's identity.

## Next exercises

1. Add validation tracking and early stopping.
2. Wire the validation-aware helper into the CLI and save a model checkpoint.
3. Try a real grouped dataset with clear consent and documentation.
