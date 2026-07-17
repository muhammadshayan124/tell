"""Linguistic features with a specific citation behind each one -- this file
exists to make it easy to check "is this a real, published cue, or did I
just make it up," not to maximize feature count.

Primary sources:
  - Newman, M. L., Pennebaker, J. W., Berry, D. S., & Richards, J. M. (2003).
    "Lying words: Predicting deception from linguistic styles." Personality
    and Social Psychology Bulletin, 29(5), 665-675.
    -> first-person singular pronouns (down in deceptive text), negative
       emotion words (up), cognitive-complexity / exclusive words (down).
  - DePaulo, B. M., Lindsay, J. J., Malone, B. E., Muhlenbruck, L.,
    Charlton, K., & Cooper, H. (2003). "Cues to deception." Psychological
    Bulletin, 129(1), 74-118.
    -> statement length as a (small) cue; the meta-analytic effect sizes in
       this paper are the reason this whole feature set should be read as
       "weakly informative," not "diagnostic."
  - Hauch, V., Blandon-Gitlin, I., Masip, J., & Sporer, S. L. (2015). "A
    meta-analysis of linguistic cues to deception." Personality and Social
    Psychology Review, 19(4), 307-343.
    -> confirms most individual linguistic cues have small effect sizes
       (Cohen's d mostly < 0.3) and are inconsistent across studies -- the
       reason this module's docstring keeps saying "weak signal."

Sentiment/emotion valence uses VADER (Hutto & Gilbert, 2014) rather than a
hand-rolled emotion lexicon, since it's a validated, freely-licensed tool
built for exactly this (lexicon + rule-based sentiment intensity), unlike
LIWC which is proprietary and can't be redistributed or reimplemented here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_WORD_RE = re.compile(r"[a-zA-Z']+")

# Newman et al. (2003), Table 1 pronoun categories.
_FIRST_PERSON_SINGULAR = {"i", "me", "my", "mine", "myself"}
_FIRST_PERSON_PLURAL = {"we", "us", "our", "ours", "ourselves"}
_OTHER_REFERENCE = {
    "he",
    "him",
    "his",
    "she",
    "her",
    "hers",
    "they",
    "them",
    "their",
    "theirs",
    "it",
    "its",
}

# Newman et al.'s "exclusive words" -- differentiating one thing from
# another, associated with more cognitively complex (truthful) narratives.
_EXCLUSIVE_WORDS = {"but", "except", "without", "unless", "although", "however"}

# Filler/disfluency tokens -- these transcripts (Mechanical-Turk transcribed,
# per the dataset's own README) explicitly preserve "um", "ah", "uh" and
# ellipsis for pauses, which is unusually good ground truth for this feature
# compared to most text corpora.
_FILLERS = {"um", "umm", "uh", "uhh", "ah", "ahh", "er", "erm"}

# Hedges/qualifiers -- tentative language, studied (with mixed results) as a
# marker of uncertainty rather than confident recall.
_HEDGES = {
    "maybe",
    "perhaps",
    "probably",
    "possibly",
    "sort",
    "kind",
    "guess",
    "think",
    "believe",
    "seem",
    "seems",
    "somewhat",
}

_sentiment_analyzer = SentimentIntensityAnalyzer()


@dataclass(frozen=True)
class TextFeatures:
    word_count: int
    first_person_singular_ratio: float
    first_person_plural_ratio: float
    other_reference_ratio: float
    exclusive_word_ratio: float
    filler_ratio: float
    hedge_ratio: float
    negative_emotion_score: float
    ellipsis_count: int


def extract_text_features(transcript: str) -> TextFeatures:
    words = [w.lower() for w in _WORD_RE.findall(transcript)]
    word_count = len(words)
    n = max(word_count, 1)  # avoid division by zero on an empty transcript

    def ratio(vocabulary: set[str]) -> float:
        return sum(1 for w in words if w in vocabulary) / n

    sentiment = _sentiment_analyzer.polarity_scores(transcript)

    return TextFeatures(
        word_count=word_count,
        first_person_singular_ratio=ratio(_FIRST_PERSON_SINGULAR),
        first_person_plural_ratio=ratio(_FIRST_PERSON_PLURAL),
        other_reference_ratio=ratio(_OTHER_REFERENCE),
        exclusive_word_ratio=ratio(_EXCLUSIVE_WORDS),
        filler_ratio=ratio(_FILLERS),
        hedge_ratio=ratio(_HEDGES),
        negative_emotion_score=sentiment["neg"],
        ellipsis_count=transcript.count("..."),
    )


FEATURE_NAMES = [
    "word_count",
    "first_person_singular_ratio",
    "first_person_plural_ratio",
    "other_reference_ratio",
    "exclusive_word_ratio",
    "filler_ratio",
    "hedge_ratio",
    "negative_emotion_score",
    "ellipsis_count",
]


def features_to_vector(features: TextFeatures) -> list[float]:
    return [getattr(features, name) for name in FEATURE_NAMES]
