from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import pyaudio
import wave

def Record():
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 16000
    RECORD_SECONDS = 5
    WAVE_OUTPUT_FILENAME = "output.wav"

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

    print("Recording...")

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Done recording.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()


def MFCCAnalyze():
    print("Starting analyzing...")

    (rate, sig) = wav.read("output.wav")
    mfcc_feat = mfcc(sig, rate)
    fbank_feat = logfbank(sig, rate)

    print("Done analyzing. \nYour features:")
    print(fbank_feat[1:3,:])
    print("\nAnd MFCC:\n")
    print(mfcc_feat[1:3,:])


if __name__ == "__main__":
    Record()
    MFCCAnalyze()
