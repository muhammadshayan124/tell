from tell.features.text import extract_text_features


def test_first_person_singular_ratio_counts_i_me_my():
    features = extract_text_features("I told my friend that I would help her")
    # "I", "my", "I" out of 9 words -> 3/9
    assert round(features.first_person_singular_ratio, 3) == round(3 / 9, 3)


def test_other_reference_ratio_counts_third_person_pronouns():
    features = extract_text_features("She said he told them about it")
    # "She", "he", "them", "it" out of 7 words
    assert round(features.other_reference_ratio, 3) == round(4 / 7, 3)


def test_exclusive_word_ratio_counts_differentiating_conjunctions():
    features = extract_text_features("I went but she stayed although it was late")
    # I, went, but, she, stayed, although, it, was, late = 9 words; "but", "although" match.
    assert round(features.exclusive_word_ratio, 3) == round(2 / 9, 3)


def test_filler_ratio_counts_disfluencies():
    features = extract_text_features("um I think uh it was fine")
    # um, I, think, uh, it, was, fine = 7 words; "um", "uh" match.
    assert round(features.filler_ratio, 3) == round(2 / 7, 3)


def test_hedge_ratio_counts_tentative_language():
    features = extract_text_features("maybe it was probably fine I guess")
    # maybe, it, was, probably, fine, I, guess = 7 words; "maybe", "probably", "guess" match.
    assert round(features.hedge_ratio, 3) == round(3 / 7, 3)


def test_ellipsis_count():
    features = extract_text_features("well... I mean... it happened")
    assert features.ellipsis_count == 2


def test_negative_emotion_score_is_higher_for_clearly_negative_text():
    negative = extract_text_features("I hate this, it was terrible and awful")
    positive = extract_text_features("I love this, it was wonderful and great")
    assert negative.negative_emotion_score > positive.negative_emotion_score


def test_word_count():
    features = extract_text_features("one two three four five")
    assert features.word_count == 5


def test_empty_transcript_does_not_raise():
    features = extract_text_features("")
    assert features.word_count == 0
    assert features.first_person_singular_ratio == 0.0
