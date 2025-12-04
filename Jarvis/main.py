import speech_recognition as sr
import webbrowser
import pyttsx3
import requests
import os
from openai import OpenAI
import musicLibrary

# ---------- CONFIG ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Store in environment variable
NEWS_API_KEY = os.getenv("NEWSAPI_KEY")       # Store in environment variable
print("OPENAI_API_KEY =", OPENAI_API_KEY)

client = OpenAI(api_key=OPENAI_API_KEY)

recognizer = sr.Recognizer()
engine = pyttsx3.init()

# ---------- SPEAK FUNCTION ----------
def speak(text):
    engine.say(text)
    engine.runAndWait()

# ---------- OPENAI PROCESSING ----------
def aiProcess(command):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are Jarvis, a fast, logical virtual assistant."},
                {"role": "user", "content": command}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return "There was a problem connecting to the AI."

# ---------- COMMAND HANDLER ----------
def processCommand(cmd):
    cmd = cmd.lower()

    # 1. OPEN WEBSITES -----------------------
    if "open google" in cmd:
        webbrowser.open("https://google.com")
        speak("Opening Google")

    elif "open youtube" in cmd:
        webbrowser.open("https://youtube.com")
        speak("Opening YouTube")

    elif "open facebook" in cmd:
        webbrowser.open("https://facebook.com")
        speak("Opening Facebook")

    elif "open linkedin" in cmd:
        webbrowser.open("https://linkedin.com")
        speak("Opening LinkedIn")

    # 2. PLAY MUSIC --------------------------
    elif cmd.startswith("play"):
        song = cmd.split(" ", 1)[1]
        if song in musicLibrary.music:
            webbrowser.open(musicLibrary.music[song])
            speak(f"Playing {song}")
        else:
            speak(f"{song} is not in your music library.")

    # 3. NEWS API ----------------------------
    elif "news" in cmd:
        try:
            url = f"https://newsapi.org/v2/top-headlines?country=in&apiKey={NEWS_API_KEY}"
            r = requests.get(url)

            if r.status_code == 200:
                articles = r.json().get("articles", [])[:3]  # top 3
                if not articles:
                    speak("I couldn't find any news right now.")
                    return

                for article in articles:
                    speak(article["title"])
            else:
                speak("Unable to fetch the news.")

        except:
            speak("There was an error fetching the news.")

    # 4. AI FALLBACK -------------------------
    else:
        output = aiProcess(cmd)
        speak(output)

# ---------- MAIN LOOP ----------
if __name__ == "__main__":
    speak("Jarvis activated.")

    while True:
        try:
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source)
                print("Listening for wake word...")

                audio = recognizer.listen(source, timeout=4, phrase_time_limit=2)
                wake = recognizer.recognize_google(audio).lower()

                if wake == "jarvis":
                    speak("Yes?")
                    print("Jarvis active...")

                    # Listen for actual command
                    with sr.Microphone() as source2:
                        recognizer.adjust_for_ambient_noise(source2)
                        audio2 = recognizer.listen(source2, timeout=5)
                        command = recognizer.recognize_google(audio2)
                        print("Command:", command)

                        processCommand(command)

        except Exception as e:
            # Ignore errors silently to keep loop alive
            print("Error:", e)
            continue
