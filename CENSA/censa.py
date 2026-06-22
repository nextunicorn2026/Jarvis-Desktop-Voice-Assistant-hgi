import pyttsx3
import datetime
import speech_recognition as sr
import wikipedia
import webbrowser as wb
import os
import random
import pyautogui
import pyjokes
from translation import (
    parse_languages, resolve_language, translate_text, speak_in,
    save_voice_profile, voice_profile_path,
)

engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)  
engine.setProperty('rate', 150)
engine.setProperty('volume', 1)


def speak(audio) -> None:
    engine.say(audio)
    engine.runAndWait()


def time() -> None:
    """Tells the current time."""
    current_time = datetime.datetime.now().strftime("%I:%M:%S %p")
    speak("The current time is")
    speak(current_time)
    print("The current time is", current_time)


def date() -> None:
    """Tells the current date."""
    now = datetime.datetime.now()
    speak("The current date is")
    speak(f"{now.day} {now.strftime('%B')} {now.year}")
    print(f"The current date is {now.day}/{now.month}/{now.year}")


def wishme() -> None:
    """Greets the user based on the time of day."""
    speak("Welcome back, sir!")
    print("Welcome back, sir!")

    hour = datetime.datetime.now().hour
    if 4 <= hour < 12:
        speak("Good morning!")
        print("Good morning!")
    elif 12 <= hour < 16:
        speak("Good afternoon!")
        print("Good afternoon!")
    elif 16 <= hour < 24:
        speak("Good evening!")
        print("Good evening!")
    else:
        speak("Good night, see you tomorrow.")

    assistant_name = load_name()
    speak(f"{assistant_name} at your service. Please tell me how may I assist you.")
    print(f"{assistant_name} at your service. Please tell me how may I assist you.")


def screenshot() -> None:
    """Takes a screenshot and saves it."""
    img = pyautogui.screenshot()
    img_path = os.path.expanduser("~\\Pictures\\screenshot.png")
    img.save(img_path)
    speak(f"Screenshot saved as {img_path}.")
    print(f"Screenshot saved as {img_path}.")

def takecommand(language="en-in") -> str:
    """Takes microphone input and returns it as text, recognised in `language`."""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        r.pause_threshold = 1

        try:
            audio = r.listen(source, timeout=5)  # Listen with a timeout
        except sr.WaitTimeoutError:
            speak("Timeout occurred. Please try again.")
            return None

    try:
        print("Recognizing...")
        query = r.recognize_google(audio, language=language)
        print(query)
        return query.lower()
    except sr.UnknownValueError:
        speak("Sorry, I did not understand that.")
        return None
    except sr.RequestError:
        speak("Speech recognition service is unavailable.")
        return None
    except Exception as e:
        speak(f"An error occurred: {e}")
        print(f"Error: {e}")
        return None

def play_music(song_name=None) -> None:
    """Plays music from the user's Music directory."""
    song_dir = os.path.expanduser("~\\Music")
    songs = os.listdir(song_dir)

    if song_name:
        songs = [song for song in songs if song_name.lower() in song.lower()]

    if songs:
        song = random.choice(songs)
        os.startfile(os.path.join(song_dir, song))
        speak(f"Playing {song}.")
        print(f"Playing {song}.")
    else:
        speak("No song found.")
        print("No song found.")

def set_name() -> None:
    """Sets a new name for the assistant."""
    speak("What would you like to name me?")
    name = takecommand()
    if name:
        with open("assistant_name.txt", "w") as file:
            file.write(name)
        speak(f"Alright, I will be called {name} from now on.")
    else:
        speak("Sorry, I couldn't catch that.")

def load_name() -> str:
    """Loads the assistant's name from a file, or uses a default name."""
    try:
        with open("assistant_name.txt", "r") as file:
            return file.read().strip()
    except FileNotFoundError:
        return "CENSA"  # Default name


def live_translate(source, target) -> None:
    """Realtime interpreter: listen in `source`, speak the translation in `target`."""
    speak(f"Live translation on. Speak in {source['name']}, I'll reply in {target['name']}. "
          f"Say stop translating to end.")
    print(f"[Interpreter] {source['name']} -> {target['name']}")
    while True:
        text = takecommand(language=source["stt"])
        if not text:
            continue
        if any(p in text for p in ("stop translating", "stop translation", "exit translation")):
            speak("Translation mode off.")
            return
        translated = translate_text(text, target["translate"], source=source["translate"])
        print(f"[{source['name']}] {text}\n[{target['name']}] {translated}")
        speak_in(translated, target["tts"], fallback_speak=speak)


def handle_translate(query) -> None:
    """Parse a translate command and start the live interpreter."""
    source, target = parse_languages(query)
    if not target:
        speak("Which language should I translate to?")
        target = resolve_language(takecommand())
    if not source:
        speak("Which language will you speak? Defaulting to English if you stay silent.")
        spoken = takecommand()
        source = resolve_language(spoken) or resolve_language("english")
    if target:
        live_translate(source, target)
    else:
        speak("Sorry, I don't support that language yet.")


def search_wikipedia(query):
    """Searches Wikipedia and returns a summary."""
    try:
        speak("Searching Wikipedia...")
        result = wikipedia.summary(query, sentences=2)
        speak(result)
        print(result)
    except wikipedia.exceptions.DisambiguationError:
        speak("Multiple results found. Please be more specific.")
    except Exception:
        speak("I couldn't find anything on Wikipedia.")


def clone_founder_voice(audio_path=None, ref_text=None, seconds=6) -> None:
    """Capture (or import) a reference clip so OmniVoice speaks in the founder's voice.

    Saves a persistent voice profile under ~/.censa that CENSA's TTS loads
    automatically on every future run. Pass an existing clip with --audio, and
    its transcript with --text (otherwise it is auto-transcribed). A 3–10s clip
    in a quiet room gives the best clone.
    """
    r = sr.Recognizer()
    dest = os.path.join(os.path.dirname(voice_profile_path()), "founder_ref.wav")

    if audio_path:
        src_path = os.path.abspath(os.path.expanduser(audio_path))
        if not os.path.exists(src_path):
            print(f"Audio file not found: {src_path}")
            return
        dest = src_path  # reference the supplied file in place
        if not ref_text:
            try:
                with sr.AudioFile(src_path) as src:
                    ref_text = r.recognize_google(r.record(src))
            except Exception:
                ref_text = ""  # empty -> OmniVoice auto-transcribes via Whisper
    else:
        try:
            with sr.Microphone() as source:
                print(f"Recording {seconds}s reference clip — speak naturally now...")
                r.adjust_for_ambient_noise(source, duration=0.5)
                audio = r.record(source, duration=seconds)
        except Exception as e:
            print(f"Could not access microphone: {e}")
            return
        with open(dest, "wb") as f:
            f.write(audio.get_wav_data())
        if not ref_text:
            try:
                ref_text = r.recognize_google(audio)
            except Exception:
                ref_text = ""

    save_voice_profile(dest, ref_text or "")
    print(f"Founder voice profile saved -> {dest}")
    if ref_text:
        print(f"Reference transcript: {ref_text}")
    print("CENSA will now speak in your cloned voice via OmniVoice (CENSA_TTS=auto).")
    speak_in("Founder voice profile created. This is how I will sound from now on.",
             "en", fallback_speak=speak)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CENSA desktop voice assistant")
    parser.add_argument("--clone-founder-voice", action="store_true",
                        help="Record/import a reference clip so CENSA speaks in your voice (OmniVoice).")
    parser.add_argument("--audio", help="Existing reference clip (3–10s) to use instead of recording.")
    parser.add_argument("--text", help="Transcript of the reference clip (optional; auto-transcribed if omitted).")
    parser.add_argument("--seconds", type=int, default=6, help="Recording length when capturing from the mic.")
    _args = parser.parse_args()

    if _args.clone_founder_voice:
        clone_founder_voice(audio_path=_args.audio, ref_text=_args.text, seconds=_args.seconds)
        raise SystemExit(0)

    wishme()

    while True:
        query = takecommand()
        if not query:
            continue

        if "translate" in query or "interpreter" in query:
            handle_translate(query)

        elif "time" in query:
            time()
            
        elif "date" in query:
            date()

        elif "wikipedia" in query:
            query = query.replace("wikipedia", "").strip()
            search_wikipedia(query)

        elif "play music" in query:
            song_name = query.replace("play music", "").strip()
            play_music(song_name)

        elif "open youtube" in query:
            wb.open("youtube.com")
            
        elif "open google" in query:
            wb.open("google.com")

        elif "change your name" in query:
            set_name()

        elif "screenshot" in query:
            screenshot()
            speak("I've taken screenshot, please check it")

        elif "tell me a joke" in query:
            joke = pyjokes.get_joke()
            speak(joke)
            print(joke)

        elif "shutdown" in query:
            speak("Shutting down the system, goodbye!")
            os.system("shutdown /s /f /t 1")
            break
            
        elif "restart" in query:
            speak("Restarting the system, please wait!")
            os.system("shutdown /r /f /t 1")
            break
            
        elif "offline" in query or "exit" in query:
            speak("Going offline. Have a good day!")
            break
