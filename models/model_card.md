---
license: mit
tags:
  - deception-detection
  - research-artifact
  - not-for-production-decisions
---

# tell -- experimental deception-cue classifier

**This is not a validated lie detector.** See the [project README](https://github.com/muhammadshayan124/tell) for the full disclaimer. This model card exists so anyone loading this model from the Hugging Face Hub sees the same caveats as everyone using the CLI/GUI directly.

## What this model is

A logistic regression over 17 documented linguistic + vocal features (see README for citations), fit on 121 real, labeled courtroom clips (61 deceptive, 60 truthful) from the [Real-life Trial Deception Detection dataset](https://web.eecs.umich.edu/~mihalcea/downloads/RealLifeDeceptionDetection.2016.zip) (Perez-Rosas et al., 2015).

## Honest performance (10-fold cross-validation, NOT training accuracy)

- Accuracy: 66.2% +/- 10.3%
- Precision: 64.3%
- Recall: 73.8%
- F1: 68.7%

Chance is 50%. This is a real, weak, above-chance signal, consistent with the published literature's own conclusion that these cues are individually weak and inconsistent. It is NOT evidence this model can tell whether a specific person is lying.

## Known limitations

- Labels are trial verdicts, not verified ground truth about whether the person lied.
- Many clips share the same speaker/case, so the model may partly be learning to recognize who is talking rather than a generalizable cue -- cross-validation folds are not guaranteed speaker-disjoint.
- n=121 is small; the +/- 10.3% standard deviation across folds should be read as "we genuinely don't know the true accuracy any more precisely than this," not a rounding error.

## Do not use this to make a real decision about a real person.

Generated 2026-07-17 from a fixed random seed; see the `tell` repository for the full pipeline.
