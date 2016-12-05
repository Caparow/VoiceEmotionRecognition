from python_speech_features import mfcc

from sklearn import preprocessing
import cPickle
import scipy.io.wavfile as wav
import pyaudio
import wave

output_frame_num = 200

def SaveObj(path, obj):
    f = open(path, 'w')
    cPickle.dump(obj, f, -1)
    f.close()

def GetObj(path):
    with open(path, 'rb') as f:
        obj = cPickle.load(f)
    f.close()
    return obj

def Record():
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 16000
    RECORD_SECONDS = 3
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

def GetMfccVector(path):
    (rate, signal) = wav.read(path)
    mfcc_vec = mfcc(signal,rate,winlen=0.025,winstep=0.01,nfft=512,lowfreq=0,
                    highfreq=None,preemph=0.97,ceplifter=22,appendEnergy=True)
    return mfcc_vec

def Normalize(data):
    return preprocessing.normalize(data,norm='l2')

def NormAndAverage(mfcc_feat):
    feat_norm = Normalize(mfcc_feat)
    frames_num = len(feat_norm)
    start_frame = (frames_num - output_frame_num) / 2
    output_feat = []
    num = 1
    for frame in feat_norm:
        if num < start_frame:
            num += 1
        elif num < start_frame + output_frame_num:
            for feat in frame:
                output_feat.append(feat)
            num += 1
        else:
            break
    return output_feat

def TrainFilesAdding():
    tmp_e = ['a', 'd', 'f', 'h', 'n', 'sa', 'su']
    tmp_a = ['DC', 'JE', 'JK', 'KL']
    x = 16
    features_matrix = []
    sample_vector = []
    print('Extracting...')

    # extract all the features from the audio segment
    for act in tmp_a:
        path = 'AudioData/' + act + '/'
        print('Start extract features from: ' + path)
        for em in tmp_e:
            path1 = path + em
            for i in range(1, x):
                if i < 10:
                    mfcc_feat = GetMfccVector(path1 + '0' + str(i) + '.wav')
                else:
                    mfcc_feat = GetMfccVector(path1 + str(i) + '.wav')

                # adding to the features matrix
                output_feat = NormAndAverage(mfcc_feat)
                features_matrix.append(output_feat)

                # adding to the sample vector
                if em == 'a':
                    sample_vector.append('ANGRY')
                elif em == 'd':
                    sample_vector.append('DISGUST')
                elif em == 'f':
                    sample_vector.append('FEAR')
                elif em == 'h':
                    sample_vector.append('HAPPINESS')
                elif em == 'n':
                    sample_vector.append('NERVOUS')
                elif em == 'sa':
                    sample_vector.append('SADNESS')
                elif em == 'su':
                    sample_vector.append('SURPRISE')

        print('Features extracted from: ' + path + '\n')

    print('Len of feat matr.= '+str(len(features_matrix)))
    SaveObj('features_matrix.dat', features_matrix)
    print('Len of sample vect.= ' + str(len(sample_vector)))
    SaveObj('sample_vector.dat', sample_vector)
    print('\nDone.')

def MFCCAnalyze():
    print("Starting analyzing...")
    Record()
    mfcc_feat = GetMfccVector('output.wav')
    output_feat = NormAndAverage(mfcc_feat)
    print("Done analyzing. \nYour features:")
    return output_feat


if __name__ == "__main__":
    features = MFCCAnalyze()
    # Use your svm here
    # TrainFilesAdding()
    # Use pickle and extract [sample, features] from features_matrix.dat
    # and to extract [sample] from sample_vector.dat
    # ONLY AFTER TrainFilesAdding() FUNC
