import sounddevice as sd
import soundfile as sf

print("Starting microphone test...")
print("Speak for 5 seconds!")

sample_rate = 16000
duration = 5

audio = sd.rec(
    int(duration * sample_rate),
    samplerate=sample_rate,
    channels=1,
    dtype="float32"
)

sd.wait()

sf.write("test_recording.wav", audio, sample_rate)

print("Recording completed!")
print("Saved as test_recording.wav")