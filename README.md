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
# optional extras:
pip install -e ".[factcheck]"   # LLM-based fact-checking layer
pip install -e ".[whisper]"     # local speech-to-text for the GUI
pip install -e ".[gui]"         # streamlit + plotly for the GUI below
pip install -e ".[hub]"         # only needed to push a model to Hugging Face
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

## Try it: the Streamlit GUI

```bash
pip install -e ".[gui,whisper]"
python scripts/train_and_save_model.py --data-dir data/Real-life_Deception_Detection_2016
streamlit run streamlit_app.py
```

Record (or upload) a clip, transcribe it locally with Whisper (`faster-whisper`, no cloud call),
and see every step of the arithmetic: the raw feature values, a chart of each feature's
`z-score * coefficient` contribution to the logit, the intercept, the sum, and the sigmoid that
turns that sum into a number.

That final number is labeled **"model classification signal," deliberately not "chance you are
lying."** It's the identical arithmetic either way — `sigmoid(logit)` — but the second phrasing
implies a claim this model cannot back up, and the honest wording costs nothing. The model's own
real cross-validated accuracy (66.2% ± 10.3%, barely better than a coin flip with real
uncertainty) sits right next to it as a permanent reality check. The disclaimer banner at the top
is not decorative -- it says outright that this is not a lie detector before anything else loads.

The GUI also exposes the fact-checking layer (`tell.factcheck`) as a separate, optional section,
keeping "does this sound evasive" and "is this specific claim true" visually distinct, for the
same reason described above.

## The saved model, and Hugging Face

`scripts/train_and_save_model.py` fits one final `Pipeline(StandardScaler, LogisticRegression)`
on all 121 examples and saves it with [skops](https://skops.readthedocs.io/) (not raw `pickle`,
so loading it never executes arbitrary code) to `models/tell_classifier.skops`, alongside
`models/metadata.json` (the honest cross-validated numbers, not training accuracy) and
`models/model_card.md` (the same disclaimers, in Hugging-Face-model-card format). All three are
small (~25KB total) and committed to this repo directly.

```python
from tell.model import load_model, explain

pipeline = load_model("models/tell_classifier.skops")
contributions = explain(pipeline, feature_vector)  # per-feature z-score * coefficient breakdown
```

To publish the same artifact to your own Hugging Face account (not run by anything in this repo
automatically -- it needs your own token and consent):

```bash
export HF_TOKEN=hf_...
python scripts/push_to_huggingface.py --repo-id your-username/tell-deception-cues
```

## Tests

31 tests: every text feature against hand-verified word counts, every audio feature against synthetic tones/silence with known properties (including a genuine `librosa.effects.split` edge case — its silence threshold is relative to a signal's own peak, so a uniformly quiet clip has no internal contrast to split on), the full cross-validation pipeline sanity-checked against both a synthetic separable signal (must score >90%) and synthetic random labels (must stay near chance, or the harness itself has a leak), the fact-check claim-extraction/verification loop against a scripted fake provider, and the save/load/`explain()` math (the per-feature breakdown is checked to sum to the exact same logit `predict_proba` produces internally, not just to "run without crashing").

```bash
pytest -q
ruff check src tests
ruff format --check src tests
```

## Related datasets we looked at, and why they're not in this repo

Part of doing this honestly meant actually looking for more data before settling for n=121:

- **[MU3D](https://sc.lib.miamioh.edu/handle/2374.MIA/6067)** (Lloyd, E. P., Deska, J. C., Hugenberg, K., et al., 2019, "Miami University deception detection database," *Behavior Research Methods*, 51, 1101–1116) — 320 videos of 80 people telling truths and lies, fully crossed on speaker race/gender and statement valence. Free for academic use, but access requires signing an agreement and emailing the corresponding author -- not something this project can complete on a user's behalf, and video-first rather than audio/text-first.
- **[Bag-of-Lies](https://iab-rubric.org/old/index.php/bag-of-lies)** (Gupta, V., Agarwal, M., Arora, M., Chakraborty, T., Singh, R., & Vatsa, M., 2019, "Bag-of-Lies: A multimodal dataset for deception detection," *CVPR Workshops*) — video, audio, eye gaze, and EEG from a lie-detection game. Also request-gated, and its primary signal is video/EEG, which is out of scope for this project's text+audio v1.

Both are cited here so anyone extending this project starts from a real map of what exists, instead of re-discovering it.

## References

- Newman, M. L., Pennebaker, J. W., Berry, D. S., & Richards, J. M. (2003). Lying words: Predicting deception from linguistic styles. *Personality and Social Psychology Bulletin*, 29(5), 665–675.
- DePaulo, B. M., Lindsay, J. J., Malone, B. E., Muhlenbruck, L., Charlton, K., & Cooper, H. (2003). Cues to deception. *Psychological Bulletin*, 129(1), 74–118.
- Zuckerman, M., DePaulo, B. M., & Rosenthal, R. (1981). Verbal and nonverbal communication of deception. *Advances in Experimental Social Psychology*, 14, 1–59.
- Vrij, A. (2008). *Detecting Lies and Deceit: Pitfalls and Opportunities* (2nd ed.). Wiley.
- Hauch, V., Blandón-Gitlin, I., Masip, J., & Sporer, S. L. (2015). A meta-analysis of linguistic cues to deception. *Personality and Social Psychology Review*, 19(4), 307–343.
- Sporer, S. L., & Schwandt, B. (2006). Paraverbal indicators of deception: A meta-analytic synthesis. *Applied Cognitive Psychology*, 20(4), 421–446.
- Hutto, C. J., & Gilbert, E. (2014). VADER: A parsimonious rule-based model for sentiment analysis of social media text. *ICWSM*.
- Pérez-Rosas, V., Abouelenien, M., Mihalcea, R., & Burzo, M. (2015). Deception detection using real-life trial data. *Proceedings of ICMI '15*.
- Lloyd, E. P., Deska, J. C., Hugenberg, K., et al. (2019). Miami University deception detection database. *Behavior Research Methods*, 51, 1101–1116.
- Gupta, V., Agarwal, M., Arora, M., Chakraborty, T., Singh, R., & Vatsa, M. (2019). Bag-of-Lies: A multimodal dataset for deception detection. *CVPR Workshops*.

## What's deliberately out of scope for v1

- **No video or facial analysis.** Text and audio only — adding a third, even weaker-evidenced modality (micro-expressions) would dilute an already-honest scope, not strengthen it.
- **No single merged "truth score."** Deception cues and fact-checking answer different questions; see above.
- **No hosted API or service.** The CLI, the saved model, and the local Streamlit GUI are all things you run yourself -- there is no "send us your audio" endpoint, and there won't be one.
