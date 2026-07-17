# tell

> ## ⚠️ This is an experimental research artifact. It is not, and is not intended to be, a working lie detector.
> No text, audio, or behavioral cue has ever been scientifically validated as a reliable indicator of deception in an individual case. Polygraphs are inadmissible in most courts for exactly this reason. This project exists to honestly test a research question — "how far can documented, published cues actually get you, measured properly?" — and to publish the real answer, including how weak it is. **Do not use this to make a real decision about a real person.** If you are looking for a tool to detect lies, this is not it, and nothing else on the market today is either.

## What this actually is

Two independent, unrelated things live in this repo:

1. **A cue-based deception classifier** (`tell.model`, `tell.features`) — a small, fully cross-validated experiment that extracts linguistic and vocal features with a specific published citation behind each one, and reports exactly how well a plain logistic regression can classify deceptive vs. truthful speech using only those features. No claim of accuracy beyond what's measured. No video or facial analysis in v1 — text + audio only.
2. **A factual claim checker** (`tell.factcheck`) — an LLM with real web search, used to check whether specific factual claims in a statement are supported or contradicted by retrievable evidence. This part is *not* pseudoscience — checking "did this event happen on this date" against public records is a well-understood problem — but it answers a completely different question than #1 and is deliberately never merged into a single score.

A statement can be behaviorally "typical of truthful speech" *and* factually wrong, or "typical of deceptive speech" *and* factually correct. Collapsing those into one number would hide that these are two different questions being asked for two different reasons.

## The actual result (not idealized, not cherry-picked)

Run against the full 121-clip [Real-life Trial Deception Detection dataset](#dataset) (Perez-Rosas et al., 2015), 10-fold stratified cross-validation, logistic regression on 17 text+audio features:

```
n = 121 (61 deceptive, 60 truthful)
Cross-validated accuracy: 66.2% +/- 10.3%
Precision: 64.3%  Recall: 73.8%  F1: 68.7%
```

Chance is 50%. This is a real, weak, above-chance signal — consistent with the published literature's own conclusion that these cues are individually weak and inconsistent (see [Hauch et al., 2015](#references)). It is not evidence that the model can tell whether a specific person is lying. A 10-point standard deviation across folds on 121 examples means the *true* accuracy could plausibly be anywhere from the high 50s to the high 70s — that uncertainty is the point of reporting it, not a rounding error to hide.

Top feature coefficients (standardized, mean across folds — sign indicates direction, magnitude indicates how much that fold-averaged feature moved the decision):

| Feature | Coefficient | Direction (higher value →) |
|---|---:|---|
| `pitch_mean_hz` | +0.872 | deceptive |
| `rms_mean` | −0.552 | truthful |
| `negative_emotion_score` | +0.498 | deceptive |
| `filler_ratio` | −0.427 | truthful |
| `other_reference_ratio` | +0.342 | deceptive |
| `word_count` | −0.325 | truthful |
| `first_person_plural_ratio` | −0.322 | truthful |
| `pause_ratio` | −0.313 | truthful |
| `hedge_ratio` | +0.304 | deceptive |
| `exclusive_word_ratio` | +0.296 | truthful |

Reproduce this yourself — see [Running the evaluation](#running-the-evaluation) below. Nothing here is asserted without a way to check it.

## Two limitations that matter more than the headline number

1. **The labels are trial verdicts, not verified truth.** A defendant found guilty didn't necessarily lie on the stand, and one found not guilty didn't necessarily tell the truth. This is the label the original dataset ships with, and it's a debated proxy in the literature it comes from — not something this project can fix, but something every number above should be read through.
2. **Many clips share the same speaker or case.** The same defendant (e.g. one high-profile case) appears in dozens of clips. A model can partly learn to recognize *who is talking* or *which case this is*, not a generalizable deception cue, and cross-validation folds here are not guaranteed speaker-disjoint. This is a real risk of the reported number being somewhat optimistic, and it's called out here rather than left for someone else to discover.

## Why build this at all, given all of the above

Because the honest version of this idea — "here's what happens when you actually measure it, with the caveats front and center" — is a legitimate thing to be curious about, and a dishonest version of it — "AI lie detector" with a percentage and no caveats — causes real harm (see HireVue and the EU's iBorderCtrl for documented examples of exactly that). This project is the former, explicitly, permanently, and is not going to quietly become the latter.

## Install

```bash
pip install -e ".[dev]"
# optional, only needed for the fact-checking layer:
pip install -e ".[factcheck]"
```

## Dataset

Not committed to this repo — it's 248MB of third-party video and its redistribution terms aren't stated on the source page. Fetch it yourself:

```bash
python scripts/fetch_dataset.py --out data/
```

This downloads the archive from the original University of Michigan host, extracts audio tracks from each clip via a portable ffmpeg, and writes `data/Real-life_Deception_Detection_2016/manifest.csv`.

## Running the evaluation

```bash
tell evaluate --data-dir data/Real-life_Deception_Detection_2016
```

## Analyzing one clip (raw features only — deliberately no "lie probability")

```bash
tell analyze transcript.txt clip.wav
```

This prints the extracted text and audio feature values and stops there. It does not print a confidence score, because a single number would misrepresent what a 66%-accurate, non-speaker-disjoint, verdict-labeled model can actually tell you about one specific clip.

## Tests

27 tests: every text feature against hand-verified word counts, every audio feature against synthetic tones/silence with known properties (including a genuine `librosa.effects.split` edge case — its silence threshold is relative to a signal's own peak, so a uniformly quiet clip has no internal contrast to split on), the full cross-validation pipeline sanity-checked against both a synthetic separable signal (must score >90%) and synthetic random labels (must stay near chance, or the harness itself has a leak), and the fact-check claim-extraction/verification loop against a scripted fake provider.

```bash
pytest -q
ruff check src tests
ruff format --check src tests
```

## References

- Newman, M. L., Pennebaker, J. W., Berry, D. S., & Richards, J. M. (2003). Lying words: Predicting deception from linguistic styles. *Personality and Social Psychology Bulletin*, 29(5), 665–675.
- DePaulo, B. M., Lindsay, J. J., Malone, B. E., Muhlenbruck, L., Charlton, K., & Cooper, H. (2003). Cues to deception. *Psychological Bulletin*, 129(1), 74–118.
- Hauch, V., Blandón-Gitlin, I., Masip, J., & Sporer, S. L. (2015). A meta-analysis of linguistic cues to deception. *Personality and Social Psychology Review*, 19(4), 307–343.
- Sporer, S. L., & Schwandt, B. (2006). Paraverbal indicators of deception: A meta-analytic synthesis. *Applied Cognitive Psychology*, 20(4), 421–446.
- Hutto, C. J., & Gilbert, E. (2014). VADER: A parsimonious rule-based model for sentiment analysis of social media text. *ICWSM*.
- Pérez-Rosas, V., Abouelenien, M., Mihalcea, R., & Burzo, M. (2015). Deception detection using real-life trial data. *Proceedings of ICMI '15*.

## What's deliberately out of scope for v1

- **No video or facial analysis.** Text and audio only — adding a third, even weaker-evidenced modality (micro-expressions) would dilute an already-honest scope, not strengthen it.
- **No single merged "truth score."** Deception cues and fact-checking answer different questions; see above.
- **Not packaged as an API or web service.** This is a CLI research tool, not a product, on purpose.
