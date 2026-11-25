from collections.abc import AsyncIterator

import httpx
import openai

import config
import dialects

SYSTEM_PROMPT = (
    "You are Rawi (راوي), a friendly voice assistant at a live demo. People talk to you in English or in "
    "Moroccan, Egyptian or Saudi Arabic, and you hear them through speech recognition, so forgive small "
    "transcription errors.\n"
    "Start every reply with exactly one tag naming the language of your reply: [en], [ar-MA], [ar-EG] or "
    "[ar-SA]. Then a space, then the reply.\n"
    "Always reply in the language and dialect of the latest message, even if earlier messages used another "
    "one: English → [en]; Moroccan Darija → [ar-MA] in Darija; Egyptian → [ar-EG] in Egyptian; Saudi or any "
    "other Arabic → [ar-SA] in Saudi dialect.\n"
    "The tag must match the dialect you actually write: if the reply is in Darija, tag [ar-MA]; if it is in "
    "Egyptian, tag [ar-EG]. Never tag a Darija or Egyptian reply as [ar-SA], and never tag Darija as [ar-EG] "
    "or the reverse.\n"
    "Tell the dialects apart by the words you write. Moroccan Darija: دابا, ديال, كنعرف, كيفاش, علاش, مزيان, "
    "واخا, بغيت, باش, نتا, هضرة. Egyptian: إزيك, إيه, ده, النهارده, دلوقتي, كده, عايز, بتاع, أوي. Saudi: وش, "
    "ليش, أبغى, تبي, الحين, شلونك, زين, وايد. If you wrote Darija words, the tag is [ar-MA]; if you wrote "
    "Egyptian words, [ar-EG]; if Saudi, [ar-SA]. Choose the tag from the words of this reply, never by "
    "copying the tag of an earlier reply.\n"
    "If the latest message mixes Arabic and English, reply in its Arabic dialect and keep the English terms "
    "as English words.\n"
    "Write Arabic in Arabic script, the way people of that dialect write casually, never transliterated.\n"
    "Reply in 2 to 4 short spoken sentences. No markdown, lists, emojis or headings.\n"
    "Be warm and brief. Do not discuss politics or religion: say politely that you're a demo assistant and "
    "offer another topic.\n"
    "Example exchanges (use the exact tag shown):\n"
    "User: كيداير اليوم؟ Assistant: [ar-MA] لاباس، الحمد لله. كيداير نتا؟\n"
    "User: شنو أحسن وقت نزور مراكش؟ Assistant: [ar-MA] الربيع، من مارس حتى ماي، الجو معتدل.\n"
    "User: إزيك عامل إيه النهارده؟ Assistant: [ar-EG] أنا تمام الحمد لله، تحب نتكلم في إيه؟\n"
    "User: قولي أكلة مصرية سهلة. Assistant: [ar-EG] كشري بالعدس والرز، سهل وسريع."
)

NO_KEY = "DeepSeek API key missing. Add DEEPSEEK_API_KEY to .env and restart."
LLM_FAILED = "DeepSeek request failed. Try again."

OFFLINE = {
    "en": "I can't reach my reply service right now. Please check the internet connection.",
    "ar": "ما أقدر أوصل لخدمة الردود الحين. تأكد من اتصال الإنترنت.",
}


class OfflineError(Exception):
    """The DeepSeek request could not be sent (no internet)."""


class DialectTagParser:
    """Strip a leading `[ar-MA]`-style tag from a token stream.

    The tag may arrive split across chunks, carry extra spaces or the wrong case, or be absent
    altogether. `feed()` returns `(dialect, text)`: the dialect is non-`None` exactly once, on the
    call that resolves the tag (or the whole buffer, when there is no tag); after that every call
    returns `(None, chunk)`.
    """

    def __init__(self):
        self._buf = ""
        self._resolved = False
        self._strip_ws = False

    @staticmethod
    def _canonical(tag: str):
        for d in dialects.DIALECTS:
            if d.lower() == tag.lower():
                return d
        return None

    def _resolve(self, require_length: bool):
        s = self._buf.lstrip()
        if s.startswith("["):
            if "]" in s[:16]:
                idx = s.index("]")
                tag = s[1:idx].strip()
                text = s[idx + 1:].lstrip()
                dialect = self._canonical(tag) or dialects.guess_dialect(text)
                self._buf = ""
                self._resolved = True
                self._strip_ws = text == ""
                return dialect, text
            if not require_length or len(s) >= 16:
                dialect = dialects.guess_dialect(s)
                self._buf = ""
                self._resolved = True
                self._strip_ws = False
                return dialect, s
            return None, ""
        if not require_length or len(s) - s.count(" ") >= 8:
            dialect = dialects.guess_dialect(s)
            self._buf = ""
            self._resolved = True
            self._strip_ws = False
            return dialect, s
        return None, ""

    def feed(self, chunk: str):
        if self._resolved:
            if self._strip_ws:
                text = chunk.lstrip()
                if text:
                    self._strip_ws = False
                    return None, text
                return None, ""
            return None, chunk
        self._buf += chunk
        dialect, text = self._resolve(require_length=True)
        if dialect is None:
            return None, ""
        return dialect, text

    def finish(self):
        if self._resolved:
            return None, ""
        dialect, text = self._resolve(require_length=False)
        return dialect, text


def build_messages(question: str, history: list[dict]) -> list[dict]:
    return [{"role": "system", "content": SYSTEM_PROMPT}, *history[-6:], {"role": "user", "content": question}]


_deepseek: openai.AsyncOpenAI | None = None  # shared client, created on first use; closed in lifespan


def get_deepseek() -> openai.AsyncOpenAI:
    global _deepseek
    if _deepseek is None:
        _deepseek = openai.AsyncOpenAI(
            api_key=config.DEEPSEEK_API_KEY,
            base_url=config.DEEPSEEK_BASE_URL,
            timeout=httpx.Timeout(30, connect=5),
            max_retries=1,
        )
    return _deepseek


async def close_deepseek() -> None:
    global _deepseek
    if _deepseek is not None:
        client, _deepseek = _deepseek, None
        await client.close()


async def stream_turn(messages: list[dict]) -> AsyncIterator[str]:
    """Stream one DeepSeek turn, yielding text chunks as they arrive."""
    try:
        stream = await get_deepseek().chat.completions.create(
            model=config.DEEPSEEK_MODEL, messages=messages, stream=True
        )
        try:
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content
        finally:
            await stream.close()
    except openai.APIConnectionError as exc:  # APITimeoutError is a subclass
        raise OfflineError() from exc
