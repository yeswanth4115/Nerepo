import tkinter as tk
import sounddevice as sd
import soundfile as sf
import pyttsx3
import speech_recognition as sr
import os
import time


# ==========================================
# SETTINGS
# ==========================================

SAMPLE_RATE = 16000
RECORD_SECONDS = 3

WORDS = [
    ("DOG", "🐶"),
    ("BALL", "⚽"),
    ("CAT", "🐱"),
    ("CAR", "🚗"),
    ("APPLE", "🍎")
]

SAVE_FOLDER = "speech_recordings"

os.makedirs(SAVE_FOLDER, exist_ok=True)


# ==========================================
# TEXT TO SPEECH
# ==========================================

engine = pyttsx3.init()
engine.setProperty("rate", 130)


# ==========================================
# SPEECH RECOGNIZER
# ==========================================

recognizer = sr.Recognizer()


# ==========================================
# VARIABLES
# ==========================================

word_index = 0
score = 0
game_running = False


# ==========================================
# FUNCTIONS
# ==========================================

def speak_word(word):

    engine.say(word)
    engine.runAndWait()


def start_game():

    global word_index
    global score
    global game_running

    word_index = 0
    score = 0
    game_running = True

    score_label.config(text="Score: 0")

    start_button.config(state="disabled")

    show_next_word()


def show_next_word():

    global word_index

    if not game_running:
        return

    # Finished
    if word_index >= len(WORDS):

        emoji_label.config(text="🎉")
        word_label.config(text="Well Done!")

        instruction_label.config(
            text=f"Final Score: {score}/{len(WORDS)}"
        )

        start_button.config(state="normal")

        return


    word, emoji = WORDS[word_index]

    emoji_label.config(text=emoji)
    word_label.config(text=word)

    instruction_label.config(
        text="Listen carefully..."
    )

    root.update()

    # Speak after a small delay
    root.after(
        500,
        lambda: speak_and_record(word)
    )


def speak_and_record(word):

    instruction_label.config(
        text=f"🔊 Listen: {word}"
    )

    root.update()

    speak_word(word)

    # Give child time to prepare
    root.after(
        1000,
        lambda: record_child(word)
    )


def record_child(word):

    instruction_label.config(
        text=f"🎤 Now say: {word}"
    )

    root.update()

    print(f"Recording response for: {word}")

    try:

        audio = sd.rec(
            int(RECORD_SECONDS * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32"
        )

        sd.wait()

    except Exception as e:

        instruction_label.config(
            text="Microphone error"
        )

        print("Microphone error:", e)

        start_button.config(state="normal")

        return


    filename = os.path.join(
        SAVE_FOLDER,
        f"{word_index + 1}_{word}.wav"
    )

    sf.write(
        filename,
        audio,
        SAMPLE_RATE
    )

    instruction_label.config(
        text="🔍 Checking your answer..."
    )

    root.update()

    # Check speech
    root.after(
        500,
        lambda: check_speech(filename, word)
    )


def check_speech(filename, target_word):

    global word_index
    global score

    try:

        with sr.AudioFile(filename) as source:

            audio = recognizer.record(source)

        spoken_text = recognizer.recognize_google(
            audio
        )

        spoken_text = spoken_text.upper().strip()

        print("Target:", target_word)
        print("Child said:", spoken_text)


        if target_word in spoken_text:

            score += 1

            instruction_label.config(
                text="⭐ Great job!"
            )

        else:

            instruction_label.config(
                text=f"Good try! I heard: {spoken_text}"
            )


    except sr.UnknownValueError:

        instruction_label.config(
            text="I couldn't hear the word. Try again!"
        )

    except sr.RequestError:

        instruction_label.config(
            text="Speech service unavailable."
        )

    except Exception as e:

        print("Speech error:", e)

        instruction_label.config(
            text="Could not check speech."
        )


    score_label.config(
        text=f"Score: {score}"
    )

    word_index += 1

    # Move to next word after 2 seconds
    root.after(
        2000,
        show_next_word
    )


def quit_game():

    global game_running

    game_running = False

    root.destroy()


# ==========================================
# GUI
# ==========================================

root = tk.Tk()

root.title("Speech Imitation Game")

root.geometry("800x600")

root.configure(bg="white")


title_label = tk.Label(
    root,
    text="🎤 Speech Imitation Game",
    font=("Arial", 28, "bold"),
    bg="white"
)

title_label.pack(pady=25)


emoji_label = tk.Label(
    root,
    text="🐶",
    font=("Segoe UI Emoji", 80),
    bg="white"
)

emoji_label.pack(pady=10)


word_label = tk.Label(
    root,
    text="DOG",
    font=("Arial", 40, "bold"),
    bg="white"
)

word_label.pack(pady=10)


instruction_label = tk.Label(
    root,
    text="Press START to begin",
    font=("Arial", 20),
    bg="white"
)

instruction_label.pack(pady=25)


score_label = tk.Label(
    root,
    text="Score: 0",
    font=("Arial", 18, "bold"),
    bg="white"
)

score_label.pack(pady=10)


start_button = tk.Button(
    root,
    text="START",
    font=("Arial", 20, "bold"),
    command=start_game,
    padx=30,
    pady=10
)

start_button.pack(pady=15)


quit_button = tk.Button(
    root,
    text="QUIT",
    font=("Arial", 14),
    command=quit_game
)

quit_button.pack()


root.mainloop()