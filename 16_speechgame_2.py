import tkinter as tk
import sounddevice as sd
import soundfile as sf
import pyttsx3
import time
import os


# -----------------------------
# Settings
# -----------------------------
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


# -----------------------------
# Text-to-speech
# -----------------------------
engine = pyttsx3.init()

engine.setProperty("rate", 130)


# -----------------------------
# Window
# -----------------------------
root = tk.Tk()

root.title("Speech Imitation Game")

root.geometry("800x600")

root.configure(bg="white")


# -----------------------------
# Variables
# -----------------------------
word_index = 0


# -----------------------------
# Functions
# -----------------------------
def speak_word(word):

    engine.say(word)
    engine.runAndWait()


def start_game():

    global word_index

    word_index = 0

    next_word()


def next_word():

    global word_index

    if word_index >= len(WORDS):

        word_label.config(text="🎉 Great job!")
        instruction_label.config(
            text="You completed all the words!"
        )

        return

    word, emoji = WORDS[word_index]

    emoji_label.config(text=emoji)

    word_label.config(text=word)

    instruction_label.config(
        text="Listen carefully..."
    )

    root.update()

    # Speak the word
    speak_word(word)

    time.sleep(1)

    instruction_label.config(
        text=f"Now say: {word}"
    )

    root.update()

    time.sleep(1)

    # Record
    instruction_label.config(
        text="🎤 Recording... Speak now!"
    )

    root.update()

    audio = sd.rec(
        int(RECORD_SECONDS * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )

    sd.wait()

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
        text="Good job! ⭐"
    )

    root.update()

    time.sleep(1)

    word_index += 1

    next_word()


# -----------------------------
# UI
# -----------------------------

title_label = tk.Label(
    root,
    text="🎤 Speech Imitation Game",
    font=("Arial", 28, "bold"),
    bg="white"
)

title_label.pack(pady=30)


emoji_label = tk.Label(
    root,
    text="🐶",
    font=("Segoe UI Emoji", 80),
    bg="white"
)

emoji_label.pack(pady=20)


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

instruction_label.pack(pady=30)


start_button = tk.Button(
    root,
    text="START",
    font=("Arial", 20, "bold"),
    command=start_game,
    padx=30,
    pady=10
)

start_button.pack(pady=20)


root.mainloop()