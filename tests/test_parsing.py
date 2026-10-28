from batching import parsing as P


def test_split_bracketed():
    s = P.split_batch_output("[1] positive\n[2] negative\n[3] neutral", 3)
    assert s.items == ["positive", "negative", "neutral"]
    assert s.n_missing == 0 and not s.extra_keys


def test_split_multiline_and_missing():
    text = "Sure! Here are the answers:\n[1] 16 - 3 = 13\n13 * 2 = 26\n#### 26\n[3] #### 5"
    s = P.split_batch_output(text, 3)
    assert s.items[0] == "16 - 3 = 13\n13 * 2 = 26\n#### 26"
    assert s.items[1] is None and s.items[2] == "#### 5"
    assert s.unnumbered_prefix.startswith("Sure")


def test_split_fallbacks_and_extras():
    s = P.split_batch_output("```\n1. yes\n2) no\nAnswer 3: yes\n4: no\n```", 3)
    assert s.items == ["yes", "no", "yes"]
    assert s.extra_keys == [4]


def test_split_single_unnumbered():
    s = P.split_batch_output("positive", 1)
    assert s.items == ["positive"]
    s = P.split_batch_output("positive", 2)
    assert s.items == [None, None]


def test_split_inline_markers():
    s = P.split_batch_output("[1] positive [2] negative [3] neutral", 3)
    assert s.items == ["positive", "negative", "neutral"]
    s = P.split_batch_output("[1] [2] [3] [4]", 4)       # empty answers -> missing
    assert s.items == [None] * 4
    # a bracketed number inside an answer that is not followed by whitespace stays put
    s = P.split_batch_output("[1] see ref [2]; fine\n[2] ok", 2)
    assert s.items == ["see ref", "ok"] or s.items[1] == "ok"


def test_split_duplicate_overrides():
    s = P.split_batch_output("[1] yes\n[1] no", 1)
    assert s.items == ["no"] and s.duplicate_keys == [1]


def test_parse_label_longest_first_and_cue():
    assert P.parse_label("very negative", P.SST5_LABELS) == "very negative"
    assert P.parse_label("The tone is positive, but overall I'd say the answer is: negative", P.SST5_LABELS) == "negative"
    assert P.parse_label("Positive.", P.SST5_LABELS) == "positive"
    assert P.parse_label("I cannot tell", P.SST5_LABELS) is None
    assert P.parse_label("Very_Positive", P.SST5_LABELS) == "very positive"


def test_parse_number():
    assert P.parse_number("16 - 3 - 4 = 9 eggs. 9 * 2 = 18\n#### 18") == "18"
    assert P.parse_number("The answer is $1,250.") == "1250"
    assert P.parse_number("So she has 3.5 left.") == "3.5"
    assert P.parse_number("\\boxed{42}") == "42"
    assert P.parse_number("no numbers here") is None
    assert P.parse_number(None) is None


def test_parse_yes_no():
    assert P.parse_yes_no("Yes, because ...") == "yes"
    assert P.parse_yes_no("Answer: No") == "no"
    assert P.parse_yes_no("False") == "no"
    assert P.parse_yes_no("maybe") is None


def test_parse_absa_pairs():
    assert P.parse_absa_pairs('[{"term": "bread", "polarity": "positive"}]') == [{"term": "bread", "polarity": "positive"}]
    assert P.parse_absa_pairs("[{'term': 'bread', 'polarity': 'Positive'}]") == [{"term": "bread", "polarity": "positive"}]
    assert P.parse_absa_pairs('[["food", "negative"], ["staff", "neutral"]]') == [
        {"term": "food", "polarity": "negative"}, {"term": "staff", "polarity": "neutral"}]
    assert P.parse_absa_pairs("[]") == []
    assert P.parse_absa_pairs("None") == []
    assert P.parse_absa_pairs('Here: [{"term": "x", "polarity": "great"}]') is None
    assert P.parse_absa_pairs("bread: positive") is None


def test_parse_gec_and_wtq():
    assert P.parse_gec('"He go to school ."') == "He go to school ."
    assert P.parse_gec("   ") is None
    assert P.parse_wtq("Italy | France") == ["Italy", "France"]
    assert P.parse_wtq("") is None
