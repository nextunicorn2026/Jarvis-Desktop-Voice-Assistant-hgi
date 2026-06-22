"""Realtime translation for the Jarvis voice assistant.

Turns Jarvis into a live interpreter: you speak in one language, it speaks the
translation back in another. Text translation uses deep-translator (Google
backend, no API key); multilingual speech uses gTTS. The offline pyttsx3 voice
is kept as the English/fallback path.

Heavy deps (deep_translator, gtts) are imported lazily inside the functions that
need them, so this module — and its self-check — load with the stdlib alone.
"""
from __future__ import annotations

import os
import sys
import tempfile
import subprocess

# Spoken language name -> (translate code, gTTS code, STT BCP-47 code).
LANGUAGES = {
    "english":    ("en",    "en",    "en-US"),
    "hindi":      ("hi",    "hi",    "hi-IN"),
    "spanish":    ("es",    "es",    "es-ES"),
    "french":     ("fr",    "fr",    "fr-FR"),
    "german":     ("de",    "de",    "de-DE"),
    "italian":    ("it",    "it",    "it-IT"),
    "portuguese": ("pt",    "pt",    "pt-PT"),
    "russian":    ("ru",    "ru",    "ru-RU"),
    "japanese":   ("ja",    "ja",    "ja-JP"),
    "korean":     ("ko",    "ko",    "ko-KR"),
    "chinese":    ("zh-CN", "zh-CN", "zh-CN"),
    "arabic":     ("ar",    "ar",    "ar-SA"),
    "bengali":    ("bn",    "bn",    "bn-IN"),
    "tamil":      ("ta",    "ta",    "ta-IN"),
    "telugu":     ("te",    "te",    "te-IN"),
    "marathi":    ("mr",    "mr",    "mr-IN"),
    "gujarati":   ("gu",    "gu",    "gu-IN"),
    "punjabi":    ("pa",    "pa",    "pa-IN"),
    "kannada":    ("kn",    "kn",    "kn-IN"),
    "urdu":       ("ur",    "ur",    "ur-PK"),
    "dutch":      ("nl",    "nl",    "nl-NL"),
    "turkish":    ("tr",    "tr",    "tr-TR"),
    "indonesian": ("id",    "id",    "id-ID"),
    "vietnamese": ("vi",    "vi",    "vi-VN"),
    "thai":       ("th",    "th",    "th-TH"),
}


def resolve_language(name):
    """Map a spoken language name to its codes, or None if unsupported.

    Tolerant of extra words ("spanish please" -> spanish).
    """
    if not name:
        return None
    name = name.strip().lower()
    for key, (t, tts, stt) in LANGUAGES.items():
        if key in name:
            return {"name": key, "translate": t, "tts": tts, "stt": stt}
    return None


def parse_languages(query):
    """Pull (source, target) language dicts out of a spoken command.

    Handles "translate to spanish" and "translate from english to japanese".
    Source defaults to None (caller decides) when only a target is given.
    """
    q = (query or "").lower()
    source = target = None
    if " to " in q:
        target = resolve_language(q.split(" to ", 1)[1])
    if " from " in q:
        seg = q.split(" from ", 1)[1]
        source = resolve_language(seg.split(" to ")[0])
    return source, target


def translate_text(text, target, source="auto"):
    """Translate text to the `target` code. Returns the original on failure."""
    if not text:
        return text
    try:
        from deep_translator import GoogleTranslator
        return GoogleTranslator(source=source, target=target).translate(text)
    except Exception as e:  # network/lang errors must not crash the assistant
        print(f"[translate] error: {e}")
        return text


def _play(path):
    """Play an audio file cross-platform without an extra Python dependency."""
    try:
        if sys.platform == "darwin":
            subprocess.run(["afplay", path], check=False)
        elif sys.platform.startswith("win"):
            os.startfile(path)  # type: ignore[attr-defined]
        else:
            for player in ("mpg123", "ffplay", "mpv"):
                if subprocess.run(["which", player], capture_output=True).returncode == 0:
                    cmd = [player, "-nodisp", "-autoexit", path] if player == "ffplay" else [player, path]
                    subprocess.run(cmd, check=False)
                    return
            print("[play] no audio player found (install mpg123/ffmpeg)")
    except Exception as e:
        print(f"[play] error: {e}")


def speak_in(text, tts_lang, fallback_speak=None):
    """Speak `text` in `tts_lang` via gTTS; use the offline voice for English."""
    if not text:
        return
    if tts_lang == "en" and fallback_speak:
        fallback_speak(text)
        return
    try:
        from gtts import gTTS
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tmp = f.name
        gTTS(text=text, lang=tts_lang).save(tmp)
        _play(tmp)
        try:
            os.remove(tmp)
        except OSError:
            pass
    except Exception as e:
        print(f"[tts] {e}; falling back to default voice")
        if fallback_speak:
            fallback_speak(text)


if __name__ == "__main__":
    # Network-free self-check of the parsing/resolution logic.
    assert resolve_language("spanish please")["translate"] == "es"
    assert resolve_language("klingon") is None
    assert resolve_language(None) is None

    s, t = parse_languages("translate to spanish")
    assert s is None and t["name"] == "spanish", (s, t)

    s, t = parse_languages("translate from english to japanese")
    assert s["name"] == "english" and t["name"] == "japanese", (s, t)

    s, t = parse_languages("translate to klingon")
    assert t is None, t

    assert translate_text("", "es") == ""  # empty short-circuits, no network
    print("OK — language resolve + command parsing pass")
