"""Realtime translation for the CENSA voice assistant.

Turns CENSA into a live interpreter: you speak in one language, it speaks the
translation back in another. Text translation uses deep-translator (Google
backend, no API key); multilingual speech uses gTTS. The offline pyttsx3 voice
is kept as the English/fallback path.

Heavy deps (deep_translator, gtts) are imported lazily inside the functions that
need them, so this module — and its self-check — load with the stdlib alone.
"""
from __future__ import annotations

import os
import sys
import json
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


# ── Founder voice profile ────────────────────────────────────────────────────
# A saved reference clip + transcript so OmniVoice clones one specific voice.
# Lives under ~/.censa (override with CENSA_HOME, used by tests).
def profile_dir():
    base = os.environ.get("CENSA_HOME") or os.path.join(os.path.expanduser("~"), ".censa")
    os.makedirs(base, exist_ok=True)
    return base


def voice_profile_path():
    return os.path.join(profile_dir(), "voice_profile.json")


def save_voice_profile(ref_audio, ref_text=""):
    with open(voice_profile_path(), "w") as f:
        json.dump({"ref_audio": ref_audio, "ref_text": ref_text}, f, indent=2)
    return voice_profile_path()


def load_voice_profile():
    try:
        with open(voice_profile_path()) as f:
            return json.load(f)
    except Exception:
        return None


# OmniVoice (k2-fsa) — state-of-the-art zero-shot multilingual TTS, 600+
# languages, voice cloning + voice design. Model is heavy (PyTorch + HF
# weights), so it is loaded once, lazily, and cached as a process singleton.
_omni = None          # loaded OmniVoice model
_omni_failed = False  # set True after a load/generate failure so we stop retrying


def _auto_device():
    """Pick the best available torch device, falling back to CPU."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda:0"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _omnivoice_speak(text):
    """Synthesize with OmniVoice. Returns True on success, False to fall back.

    Voice selection (env, optional):
      OMNIVOICE_REF_AUDIO / OMNIVOICE_REF_TEXT  -> clone a specific voice
      OMNIVOICE_INSTRUCT  (e.g. "female, low pitch, british accent") -> design
      OMNIVOICE_DEVICE    -> override auto device (cuda:0 / mps / cpu / xpu)
    """
    global _omni, _omni_failed
    if _omni_failed:
        return False
    try:
        if _omni is None:
            import torch
            from omnivoice import OmniVoice
            device = os.environ.get("OMNIVOICE_DEVICE") or _auto_device()
            dtype = torch.float16 if device != "cpu" else torch.float32
            print(f"[omnivoice] loading model on {device} (first run downloads weights)…")
            _omni = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map=device, dtype=dtype)

        kwargs = {}
        ref = os.environ.get("OMNIVOICE_REF_AUDIO")
        ref_text = os.environ.get("OMNIVOICE_REF_TEXT")
        instruct = os.environ.get("OMNIVOICE_INSTRUCT")
        # No explicit voice in env -> fall back to the saved founder profile.
        if not ref and not instruct:
            prof = load_voice_profile()
            if prof and prof.get("ref_audio") and os.path.exists(prof["ref_audio"]):
                ref = prof["ref_audio"]
                ref_text = ref_text or prof.get("ref_text")
        if ref:
            kwargs["ref_audio"] = ref
            if ref_text:
                kwargs["ref_text"] = ref_text
        elif instruct:
            kwargs["instruct"] = instruct

        audio = _omni.generate(text=text, **kwargs)  # list of np.ndarray @ 24 kHz
        import soundfile as sf
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = f.name
        sf.write(tmp, audio[0], 24000)
        _play(tmp)
        try:
            os.remove(tmp)
        except OSError:
            pass
        return True
    except Exception as e:
        print(f"[omnivoice] unavailable ({e}); falling back to gTTS/offline")
        _omni_failed = True
        return False


def speak_in(text, tts_lang, fallback_speak=None):
    """Speak `text` aloud, best engine first.

    OmniVoice (600+ languages, high quality) -> gTTS (online) -> pyttsx3 (offline).
    Set CENSA_TTS=gtts to skip OmniVoice, or =omnivoice to require it.
    """
    if not text:
        return
    engine = os.environ.get("CENSA_TTS", "auto").lower()

    if engine in ("auto", "omnivoice"):
        if _omnivoice_speak(text):
            return
        if engine == "omnivoice":
            print("[tts] OmniVoice requested but unavailable; using fallback")

    # Fallbacks: offline voice for English, gTTS for everything else.
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

    assert _auto_device() in ("cpu", "mps", "cuda:0")  # never raises
    speak_in("", "es")  # empty text is a no-op across every engine

    os.environ["CENSA_HOME"] = tempfile.mkdtemp()  # isolate the profile test
    assert load_voice_profile() is None
    save_voice_profile("/tmp/founder_ref.wav", "this is my voice")
    prof = load_voice_profile()
    assert prof["ref_audio"] == "/tmp/founder_ref.wav" and prof["ref_text"] == "this is my voice"
    print("OK — language resolve + command parsing + tts routing + voice profile pass")
