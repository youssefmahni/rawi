"""The voice follows the dialect of the words DeepSeek wrote, not only the tag it put in front of them.

Live bug (2026-11-24): a Moroccan question got a Darija reply tagged [ar-SA], so the Saudi voice spoke it.
The fixtures below are real replies from that session.
"""
import pytest

import dialects as d
import llm
from conftest import ask, events, fake_turns, keyed  # noqa: F401

REAL_MOROCCAN = ("لاباس، الله يخليك. كنعرف نهضر معاك بالدارجة والإنجليزية على أمور عادية بحال الماكلة والسفر "
                 "والخطط ديال النهار. شنو بغيتي نهضروا عليه دابا؟")
REAL_MOROCCAN_2 = ("مزيان، وجّه رسالتك الرئيسية فجملة وحدة وبنِ السلايدات عليها. حضّر مقدمة قصيرة، ثلاث نقاط "
                   "أساسية، وخاتمة فيها الخلاصة والسؤال اللي بغيتي تجاوب عليه. جرّب تقدّمها بصوت عالي مرة قبل "
                   "الاجتماع باش تلقى الوقت والثقة.")
REAL_EGYPTIAN = "أنا تمام الحمد لله، مبسوط إنك بتتكلم معايا. إنت عامل إيه النهارده؟"
REAL_SAUDI = ("بخير يا طيب، الله يسعدك. اليوم جو حلو، لو تقدر تطلع تتمشى شوي أو تقعد مع أهلك على قهوة، "
              "بتحس بانتعاش. وش رايك؟")
REAL_SAUDI_2 = ("الرياض فيها أماكن حلوة، إذا تحب الطبيعة عندك وادي حنيفة والدرعية التاريخية للتراث. بالليل تقدر "
                "تروح بوليفارد الرياض أو المربع الجديد للعشاء والتمشي. وش جوّك أقرب، هدوء ولا فعاليات؟")
MSA = "مرحبا بك، كيف يمكنني مساعدتك اليوم؟ يسعدني أن أتحدث معك عن أي موضوع."


# ---- word-level detection ----
def test_normalize_arabic():
    assert d.normalize_arabic("إِيه") == "ايه"
    assert d.normalize_arabic("النهارده") == "النهارده"
    assert d.normalize_arabic("مـــزيان") == "مزيان"
    assert d.normalize_arabic("أبغى") == "ابغي"


@pytest.mark.parametrize("text, expected", [
    (REAL_MOROCCAN, "ar-MA"),
    (REAL_MOROCCAN_2, "ar-MA"),
    (REAL_EGYPTIAN, "ar-EG"),
    (REAL_SAUDI, "ar-SA"),
    (REAL_SAUDI_2, "ar-SA"),
])
def test_detects_real_replies(text, expected):
    assert d.detect_arabic_dialect(text) == expected


def test_modern_standard_arabic_and_other_text_have_no_dialect():
    assert d.detect_arabic_dialect(MSA) is None
    assert d.detect_arabic_dialect("Hello, how are you?") is None
    assert d.detect_arabic_dialect("") is None


def test_one_weak_marker_is_not_enough():
    assert d.detect_arabic_dialect("شنو") is None


def test_markers_next_to_arabic_punctuation_still_count():
    # live bug: the word pattern swallowed ؟ ، ؛ so "إيه؟" was one unknown token
    assert d.detect_arabic_dialect("إيه؟ ده، النهارده؛") == "ar-EG"
    assert d.detect_arabic_dialect("وش رايك؟ وش تبي،") == "ar-SA"
    assert d.detect_arabic_dialect("علاش؟ دابا،") == "ar-MA"
    assert d.arabic_dialect_scores("دابا؟")["ar-MA"] == 2


def test_marker_with_a_one_letter_prefix_still_counts():
    # "and now" / "and mine": neither token is a marker until the leading و is taken off
    assert d.detect_arabic_dialect("ودابا وديالي") == "ar-MA"


def test_scores_are_per_dialect():
    scores = d.arabic_dialect_scores(REAL_MOROCCAN)
    assert set(scores) == {"ar-MA", "ar-EG", "ar-SA"}
    assert scores["ar-MA"] > scores["ar-SA"] and scores["ar-MA"] > scores["ar-EG"]


# ---- tag vs words ----
def test_wrong_arabic_tag_is_overridden_by_the_words():
    assert d.resolve_dialect("ar-SA", REAL_MOROCCAN) == "ar-MA"
    assert d.resolve_dialect("ar-EG", REAL_SAUDI) == "ar-SA"
    assert d.resolve_dialect("ar-MA", REAL_EGYPTIAN) == "ar-EG"


def test_matching_tag_is_kept():
    for tag, text in (("ar-MA", REAL_MOROCCAN), ("ar-EG", REAL_EGYPTIAN), ("ar-SA", REAL_SAUDI)):
        assert d.resolve_dialect(tag, text) == tag


def test_tag_is_kept_when_the_words_do_not_decide():
    assert d.resolve_dialect("ar-SA", MSA) == "ar-SA"
    assert d.resolve_dialect("ar-MA", MSA) == "ar-MA"
    assert d.resolve_dialect("ar-SA", "شنو بغيت") == "ar-SA"  # two weak Moroccan hints, one shared with Gulf


def test_tag_is_kept_when_both_dialects_show_up_about_equally():
    assert d.resolve_dialect("ar-SA", "وش تبي؟ شنو بغيتي؟") == "ar-SA"


def test_english_tag_on_arabic_text_becomes_arabic():
    assert d.resolve_dialect("en", "مرحبا كيف حالك اليوم") == "ar-SA"
    assert d.resolve_dialect("en", REAL_MOROCCAN) == "ar-MA"


def test_arabic_tag_on_english_text_becomes_english():
    assert d.resolve_dialect("ar-SA", "Sure, here you go.") == "en"


# ---- the user's own words as a backup when the reply is neutral ----
def test_question_decides_when_the_reply_words_do_not():
    reply = "أنا مساعد صوتي للديمو، أقدر أتكلم معاك وأرد على أسئلتك بسرعة وبساطة."   # real: neutral, tagged ar-SA
    assert d.resolve_dialect("ar-SA", reply, "إنت بتعرف تعمل إيه؟") == "ar-EG"
    assert d.resolve_dialect("ar-SA", "أسهل حاجة هي سلطة مغربية.", "عطيني فكرة على شي ماكلة مغربية ساهلة") == "ar-MA"


def test_question_is_ignored_when_the_reply_words_back_the_tag():
    assert d.resolve_dialect("ar-SA", REAL_SAUDI, "إزيك عامل إيه النهارده؟") == "ar-SA"


def test_question_without_dialect_words_changes_nothing():
    assert d.resolve_dialect("ar-SA", MSA, "كيف حالك اليوم؟") == "ar-SA"
    assert d.resolve_dialect("ar-SA", MSA, "") == "ar-SA"


def test_question_also_picks_the_dialect_of_an_arabic_reply_tagged_english():
    assert d.resolve_dialect("en", "مرحبا كيف حالك اليوم", "إزيك عامل إيه النهارده؟") == "ar-EG"


# ---- the live misses of 2026-11-24 (the first sentence of each reply) ----
@pytest.mark.parametrize("sentence, expected", [
    ("أحسن وقت هو الربيع، من مارس حتى ماي، الجو كيكون معتدل وزوين.", "ar-MA"),
    ("السما كتبان زرقا حيت الجزيئات ديال الهواء كتفرق الضو الأزرق.", "ar-MA"),
    ("أسهل حاجة هي سلطة مغربية، وكاين ماكلة أخرى بحال الحريرة.", "ar-MA"),
    ("أحسن مكان تزوره في القاهرة هو خان الخليلي، فيه جو قديم وحياة أوي.", "ar-EG"),
])
def test_first_sentences_that_the_forty_character_check_missed(sentence, expected):
    assert d.resolve_dialect("ar-SA", sentence) == expected


def test_english_stays_english_even_with_an_arabic_word():
    assert d.resolve_dialect("en", "Hello there") == "en"
    assert d.resolve_dialect("en", "Sure, the اجتماع is tomorrow at nine") == "en"


# ---- the gate that holds the dialect until the first sentence is in ----
def test_gate_holds_past_the_minimum_until_a_sentence_ends():
    g = d.DialectGate(min_chars=10)
    assert g.push("ar-SA", "هلا") == (None, [])
    assert g.push(None, " والله يا طيب") == (None, [])          # long enough, but the sentence is not over
    assert g.push(None, "، كيف حالك؟") == (None, [])            # a sentence end is only a sentence end once a space follows it
    assert g.push(None, " ") == ("ar-SA", ["هلا", " والله يا طيب", "، كيف حالك؟", " "])
    assert g.push(None, "وش") == (None, ["وش"])


def test_gate_waits_for_a_second_sentence_when_the_first_is_very_short():
    g = d.DialectGate(min_chars=40)
    assert g.push("en", "Yes. ") == (None, [])
    assert g.push(None, "Here is a longer second sentence about it. ") == (
        "en", ["Yes. ", "Here is a longer second sentence about it. "])


def test_gate_releases_at_the_cap_even_without_a_sentence_end():
    g = d.DialectGate(min_chars=5, max_chars=20)
    assert g.push("ar-SA", "س" * 25) == ("ar-SA", ["س" * 25])


def test_gate_checks_the_tag_against_the_words_it_held():
    g = d.DialectGate()
    assert g.push("ar-SA", "") == (None, [])
    assert g.push(None, REAL_MOROCCAN + " ") == ("ar-MA", [REAL_MOROCCAN + " "])


def test_gate_uses_the_question_when_the_reply_words_do_not_decide():
    g = d.DialectGate(question="عطيني فكرة على شي ماكلة مغربية ساهلة")
    assert g.push("ar-SA", "أسهل حاجة هي سلطة مغربية، غير الطماطم والخيار. ") == (
        "ar-MA", ["أسهل حاجة هي سلطة مغربية، غير الطماطم والخيار. "])


def test_gate_flush_releases_a_short_reply():
    g = d.DialectGate()
    g.push("ar-SA", "هلا")
    assert g.flush() == ("ar-SA", ["هلا"])
    assert g.flush() == (None, [])


def test_gate_flush_with_nothing_received_releases_nothing():
    assert d.DialectGate().flush() == (None, [])


def test_gate_flush_without_a_tag_guesses_from_the_script():
    g = d.DialectGate()
    g.push(None, "مرحبا")
    assert g.flush() == ("ar-SA", ["مرحبا"])


def test_gate_ignores_empty_pieces():
    g = d.DialectGate(min_chars=3)
    assert g.push("en", "") == (None, [])
    assert g.push(None, "Hi there. ") == ("en", ["Hi there. "])


# ---- the prompt teaches the markers ----
def test_prompt_lists_the_words_that_tell_the_dialects_apart():
    p = llm.SYSTEM_PROMPT
    for word in ("دابا", "ديال", "كنعرف", "إزيك", "النهارده", "دلوقتي", "وش", "الحين", "أبغى"):
        assert word in p, word
    assert "tag" in p and "actually" in p


# ---- the whole flow ----
def test_ask_overrides_a_wrong_tag_before_any_text(monkeypatch, keyed):
    fake_turns(monkeypatch, ["[ar-SA] ", REAL_MOROCCAN[:30], REAL_MOROCCAN[30:]])
    ev = events(ask(question="لاباس عليك؟ شنو كتعرف دير؟"))
    assert ev[0] == ("dialect", {"dialect": "ar-MA"})
    assert "".join(data["text"] for kind, data in ev if kind == "delta") == REAL_MOROCCAN
    assert [kind for kind, _ in ev].count("dialect") == 1 and ev[-1] == ("done", {})


def test_ask_uses_the_question_for_a_neutral_reply(monkeypatch, keyed):
    fake_turns(monkeypatch, ["[ar-SA] ", "أسهل حاجة هي سلطة مغربية، غير الطماطم والخيار والبصل والزيت والليمون."])
    ev = events(ask(question="عطيني فكرة على شي ماكلة مغربية ساهلة"))
    assert ev[0] == ("dialect", {"dialect": "ar-MA"}) and ev[-1] == ("done", {})


def test_ask_waits_for_the_first_sentence_before_it_decides(monkeypatch, keyed):
    # the Moroccan words only show up after the first 40 characters
    first = "أحسن وقت هو الربيع، من مارس حتى ماي، الجو كيكون معتدل وزوين. "
    fake_turns(monkeypatch, ["[ar-SA] ", first[:41], first[41:], "حتى الخريف زوين."])
    ev = events(ask(question="شنو هو أحسن وقت باش نزور مراكش؟"))
    assert ev[0] == ("dialect", {"dialect": "ar-MA"})
    assert "".join(data["text"] for kind, data in ev if kind == "delta") == first + "حتى الخريف زوين."


def test_ask_keeps_a_correct_tag(monkeypatch, keyed):
    fake_turns(monkeypatch, ["[ar-EG] ", REAL_EGYPTIAN])
    ev = events(ask(question="إزيك؟"))
    assert ev[0] == ("dialect", {"dialect": "ar-EG"}) and ev[-1] == ("done", {})


def test_ask_short_reply_sends_dialect_then_text(monkeypatch, keyed):
    fake_turns(monkeypatch, ["[ar-MA] لاباس"])
    assert events(ask(question="لاباس؟")) == [
        ("dialect", {"dialect": "ar-MA"}), ("delta", {"text": "لاباس"}), ("done", {})]


def test_ask_error_flushes_the_text_it_was_holding(monkeypatch, keyed):
    fake_turns(monkeypatch, ["[ar-MA] لاباس", llm.OfflineError()])
    ev = events(ask(question="لاباس؟"))
    assert ev[:2] == [("dialect", {"dialect": "ar-MA"}), ("delta", {"text": "لاباس"})]
    assert ev[2][0] == "error" and ev[2][1]["kind"] == "offline"


def test_ask_error_before_any_reply_sends_no_dialect(monkeypatch, keyed):
    fake_turns(monkeypatch, [llm.OfflineError()])
    assert [kind for kind, _ in events(ask(question="hi"))] == ["error"]
