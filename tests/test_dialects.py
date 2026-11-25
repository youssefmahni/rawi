import dialects as d


def test_dialect_codes():
    assert d.DIALECTS == ("en", "ar-MA", "ar-EG", "ar-SA") and d.DEFAULT_ARABIC == "ar-SA"


def test_has_arabic():
    assert d.has_arabic("وش أخبارك") and d.has_arabic("meeting بكرة") and not d.has_arabic("Hello 3.5!")


def test_guess_and_language():
    assert d.guess_dialect("كيداير") == "ar-SA" and d.guess_dialect("hi") == "en"
    assert d.language_of("ar-MA") == "ar" and d.language_of("en") == "en"
