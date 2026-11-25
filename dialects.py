import re
from typing import Literal

DIALECTS: tuple[str, ...] = ("en", "ar-MA", "ar-EG", "ar-SA")
DEFAULT_ARABIC = "ar-SA"

Dialect = Literal["en", "ar-MA", "ar-EG", "ar-SA"]


def has_arabic(text: str) -> bool:
    """True when any char is in U+0600–U+06FF or U+0750–U+077F."""
    return any("\u0600" <= ch <= "\u06FF" or "\u0750" <= ch <= "\u077F" for ch in text)


def language_of(dialect: str) -> Literal["en", "ar"]:
    return "ar" if dialect.startswith("ar") else "en"


def guess_dialect(text: str) -> Dialect:
    return DEFAULT_ARABIC if has_arabic(text) else "en"


# ---- telling the Arabic dialects apart by the words that were actually written ----
# DeepSeek tags every reply with its dialect, and the tag picks the voice. The tag is sometimes wrong
# (a Darija reply tagged [ar-SA] was spoken by the Saudi voice), so the words get the last say.
_DIACRITICS = re.compile("[ً-ٰٟـ]")  # harakat, superscript alef, tatweel
_FOLD = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي"})
# Arabic letters only: the block U+0600–U+06FF also holds ، ؛ ؟ and the Arabic digits and full stop,
# which must end a word, not become part of it. (Tatweel U+0640 is gone after normalize_arabic.)
_ARABIC_WORD = re.compile("[ء-ؿف-يٱ-ۓ]+")


def normalize_arabic(text: str) -> str:
    """Drop diacritics and tatweel, fold the alef forms and alef maqsura, so spelling variants match."""
    return _DIACRITICS.sub("", text).translate(_FOLD)


def _markers(strong: str, weak: str = "") -> dict[str, int]:
    out = {normalize_arabic(w): 1 for w in weak.split()}
    out.update({normalize_arabic(w): 2 for w in strong.split()})
    return out


# Strong (2 points): words that belong to one dialect among these three. Weak (1 point): words that are
# common there but shared with a neighbouring dialect (شنو is Moroccan and Gulf, بغيت is Moroccan and Najdi).
MARKERS: dict[str, dict[str, int]] = {
    "ar-MA": _markers(
        "ديال ديالي ديالك ديالو ديالها ديالنا ديالهم دابا كنعرف كتعرف كيعرف كنهضر كتهضر كيهضر نهضر نهضرو نهضروا "
        "هضرة مزيان مزيانة بزاف غادي واخا بغيتي كاين ماكاينش علاش كيفاش خويا بحال الماكلة كلشي باش بلاصة هادشي "
        "نتا نتي واش فاش ديما شحال زوين حيت عاود لاباس بسلامة كندير كتدير كيدير كيداير كتداير كنقدر كتقدر كيقدر "
        "كنبغي كتبغي كيبغي دغيا ماكلة كيكون كتكون كنكون كتبان كيبان كتفرق كنعاون كنعاونك حوايج",
        "شنو بغيت صافي ماشي هاد هادي"),
    "ar-EG": _markers(
        "ازيك ازاي ايه النهارده دلوقتي دلوقت كده ده بتاع بتاعي بتاعك بتاعتي بتاعنا عايز عايزة عاوز عاوزة اوي "
        "برضه برضو امتى جامد معايا",
        "دي دا حاجة فين لسه مش طب"),
    "ar-SA": _markers(
        "وش ابغي تبي تبغي يبي ابي نبي الحين شلونك شخبارك وايد حياك ودي ابشر يبه هاليوم هالمكان هالشي هالحين شلون "
        "اسوي تسوي يسوي نسوي",
        "ليش ايش زين كذا شوي شنو وين"),
}


def arabic_dialect_scores(text: str) -> dict[str, int]:
    """Points per dialect from the marker words in the text. A leading و or ف is tried off, too."""
    scores = {dialect: 0 for dialect in MARKERS}
    for token in _ARABIC_WORD.findall(normalize_arabic(text)):
        candidates = [token]
        if len(token) > 3 and token[0] in "وف":
            candidates.append(token[1:])
        for dialect, markers in MARKERS.items():
            scores[dialect] += max((markers.get(c, 0) for c in candidates), default=0)
    return scores


def detect_arabic_dialect(text: str) -> Dialect | None:
    """The dialect the words point to, or None when they don't decide (at least 2 points and a clear lead)."""
    scores = arabic_dialect_scores(text)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    (best, top), (_, runner_up) = ranked[0], ranked[1]
    return best if top >= 2 and top > runner_up else None  # type: ignore[return-value]


def _script_counts(text: str) -> tuple[int, int]:
    arabic = sum(1 for ch in text if "؀" <= ch <= "ۿ" or "ݐ" <= ch <= "ݿ")
    latin = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    return arabic, latin


def resolve_dialect(tag: str, text: str, question: str = "") -> Dialect:
    """The dialect to speak: the tag, unless the words clearly say otherwise.

    1. An Arabic dialect only changes when another one leads the tagged one by at least 2 points, so a thin
       or shared hint (one Moroccan word in a Saudi reply) never overrides the tag.
    2. When the reply's own words do not back the dialect it ended up with (a neutral reply), the
       dialect of the question wins if its words are clear: the user is answered in their own dialect."""
    arabic, latin = _script_counts(text)
    if tag == "en":
        if arabic <= latin:
            return "en"
        result: str = detect_arabic_dialect(text) or DEFAULT_ARABIC  # an Arabic reply that was tagged English
    elif arabic == 0 and latin > 0:  # tagged Arabic, but the reply is English
        return "en"
    else:
        scores = arabic_dialect_scores(text)
        best = max(scores, key=scores.get)
        result = best if best != tag and scores[best] >= 2 and scores[best] - scores.get(tag, 0) >= 2 else tag
    if question and arabic_dialect_scores(text).get(result, 0) < 2:
        return detect_arabic_dialect(question) or result  # type: ignore[return-value]
    return result  # type: ignore[return-value]


_SENTENCE_END = re.compile(r"[.!?؟…](?=\s)")  # a terminator only counts once a space follows it ("3.5" is not one)


class DialectGate:
    """Holds the dialect (and the text that follows it) until the first sentence is in, then checks the tag
    against the words that were actually written.

    The Arabic first sentence is about 63 characters long (at most 104 in 160 live replies), and the page
    cannot speak a sentence before it is complete anyway, so waiting for it adds no delay to the first audio.
    It releases when at least `min_chars` have arrived and a sentence has ended, or at `max_chars` whatever
    happens, or on `flush()` (end of reply, or an error).

    `push(tag, piece)` takes what the tag parser returned for one chunk. It returns `(dialect, pieces)`:
    nothing while it holds, then once, when it releases, the checked dialect and every held piece;
    after that `(None, [piece])`."""

    def __init__(self, min_chars: int = 40, max_chars: int = 120, question: str = ""):
        self._min = min_chars
        self._max = max_chars
        self._question = question
        self._tag: str | None = None
        self._pieces: list[str] = []
        self._released = False

    def _release(self) -> tuple[Dialect, list[str]]:
        text = "".join(self._pieces)
        tag = self._tag or guess_dialect(text)
        pieces, self._pieces, self._released = self._pieces, [], True
        return resolve_dialect(tag, text, self._question), pieces

    def push(self, tag: str | None, piece: str) -> tuple[Dialect | None, list[str]]:
        if self._released:
            return None, ([piece] if piece else [])
        if tag:
            self._tag = tag
        if piece:
            self._pieces.append(piece)
        text = "".join(self._pieces)
        size = len(text.strip())
        if size >= self._max or (size >= self._min and _SENTENCE_END.search(text)):
            return self._release()
        return None, []

    def flush(self) -> tuple[Dialect | None, list[str]]:
        if self._released or (self._tag is None and not self._pieces):
            return None, []
        return self._release()
