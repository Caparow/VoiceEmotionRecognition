from python_speech_features import mfcc

from sklearn import preprocessing
import cPickle
import numpy as np
from sklearn import svm
import scipy.io.wavfile as wav
from sklearn.externals import joblib
import pyaudio
import wave

#libs for testing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit

OUTPUT_FRAMES_NUM = 1600
FRAME_LEN = 0.0025
FRAME_STEP = 0.001
RECORD_SECONDS = 3

def SaveObj(path, obj):
    f = open(path, 'wb')
    cPickle.dump(obj, f, -1)
    f.close()

def GetObj(path):
    with open(path, 'rb') as f:
        obj = cPickle.load(f)
    return obj

def Record(path):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44000
    WAVE_OUTPUT_FILENAME = path

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
    mfcc_vec = mfcc(signal,rate,winlen=FRAME_LEN,winstep=FRAME_STEP,nfft=512,lowfreq=0,numcep=26,
                    highfreq=None,preemph=0.97,ceplifter=22,appendEnergy=True)
    return mfcc_vec

def Normalize(data):
    return preprocessing.normalize(data,norm='l2')

def NormAndAverage(mfcc_feat):
    feat_norm = Normalize(mfcc_feat)
    frames_num = len(feat_norm)
    start_frame = (frames_num - OUTPUT_FRAMES_NUM) / 2
    output_feat = []
    num = 1
    for frame in feat_norm:
        if num < start_frame:
            num += 1
        elif num < start_frame + OUTPUT_FRAMES_NUM:
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
    f = 0
    features_matrix = np.array((), dtype=np.float64)
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
                if f == 0:
                    features_matrix = np.concatenate((features_matrix, output_feat))
                    f += 1
                else:
                    features_matrix = np.vstack((features_matrix, output_feat))


                # adding to the sample vector
                if em == 'a':
                    sample_vector.append("ANGRY")
                elif em == 'd':
                    sample_vector.append("DISGUSTING")
                elif em == 'f':
                    sample_vector.append("FEAR")
                elif em == 'h':
                    sample_vector.append("HAPPINESS")
                elif em == 'n':
                    sample_vector.append("NEUTRAL")
                elif em == 'sa':
                    sample_vector.append("SADNESS")
                elif em == 'su':
                    sample_vector.append("SURPRISE")

        print('Features extracted from: ' + path + '\n')

    print('Len of feat matr.= ' + str(len(features_matrix)))
    SaveObj('X.dat', features_matrix)
    print('Len of sample vect.= ' + str(len(sample_vector)))
    SaveObj('Y.dat', sample_vector)
    print('\nDone.')

def MFCCAnalyze(path, rflag):
    # path - path for your file
    # tflag - '1' for record from micro
    #         else extract features from file 'path'

    if rflag == 1:
        Record(path)  # comment this and send path as your testing file
    print("Starting extracting features...")
    mfcc_feat = GetMfccVector(path)
    output_feat = NormAndAverage(mfcc_feat)
    print("Done extracting.")
    return np.array(output_feat).reshape(1, -1)

def TrainModel(path, tflag):
    #path - path for your file
    #tflag - '1' for extract features from files
    #        else train with files that already exist

    if tflag == 1:
        TrainFilesAdding()

    print('Training a model...')

    X = GetObj('X.dat')
    y = GetObj('Y.dat')
    
    clf = svm.LinearSVC(C = 1000.0, multi_class = 'ovr')
    clf.fit(X, y)

    joblib.dump(clf, path)
    print ('Done.')

def MakingPrediction(path, clf):
    features = MFCCAnalyze(path, 1)

    print('Making a prediction...')
    res = clf.predict(features)
    print "And this is " + str(res)[2:len(res) - 3] + " voice."

def TestModel(path):
    X = GetObj('X.dat')
    y = GetObj('Y.dat')

    print('Start testing model...')

    clf = svm.LinearSVC(C=1000.0, multi_class='ovr')
    cv = StratifiedShuffleSplit(n_splits=20, test_size=0.05, random_state=0)
    error = np.mean(cross_val_score(clf, X, y, cv = cv))

    print 'Cross-val error = ' + str(error)
    print('Training a model...')

    clf.fit(X, y)
    joblib.dump(clf, path)

    print ('Done.')



