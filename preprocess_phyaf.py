
import numpy as np 
import matplotlib.pyplot as plt
import scipy.io
import wfdb
from python_speech_features import mfcc, fbank, logfbank, ssc, delta
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import spectrogram, resample
import numpy as np
import matplotlib.pyplot as plt
from PyEMD import EMD
import scipy
from python_speech_features import mfcc, logfbank, delta
from sklearn.preprocessing import MinMaxScaler

s = np.load('s.npy')

def statistical_features(s):
    return [np.mean(s), np.std(s), scipy.stats.entropy(s)]

def norm(a):
    return (a - np.min(a)) / np.ptp(a)

def welch_power(s, sample_rate=300.0, fbins=list(range(0,150,10))):
    f, pxx = scipy.signal.welch(s, sample_rate)
    fbin_power = []
    for i in range(len(fbins)-1):
        ids = np.where((f >= fbins[i]) & (f < fbins[i+1]))
        fbin_power.append(np.trapz(s[ids]))
    return fbin_power

def mfcc_2d_features(s, sample_rate=300.0):
    spec = mfcc(s, samplerate=sample_rate, numcep=26, winstep=0.015).T
    spec_lfb = logfbank(s, samplerate=sample_rate, nfilt=26, winstep=0.015).T
    del_ = delta(spec, 1)
    del_d_ = delta(spec, 2)

    features = [spec, del_, del_d_, spec_lfb]
    scalar = MinMaxScaler((0, 1))

    features = [scalar.fit_transform(x) for x in features]
    features = [np.expand_dims(f, axis=0) for f in features]

    return np.concatenate(features, axis=0)

def emd_features(s, sample_rate=300.0):
    t = np.linspace(0, s.shape[0]/sample_rate, s.shape[0])
    imf = EMD().emd(s, t)
    return imf

def spectrogram_features(s, sample_rate):
    fet_maps = []
    for mode in  ['psd', 'complex', 'magnitude', 'angle', 'phase']:
        f, t, Sxx = scipy.signal.spectrogram(s, sample_rate, nperseg=128, mode=mode)
        if mode is 'complex':
            fet_maps += [Sxx.real, Sxx.imag]
        else:
            fet_maps += [Sxx]
    scalar = MinMaxScaler((0, 1))
    features = [scalar.fit_transform(x) for x in fet_maps]
    features = [np.expand_dims(f, axis=0) for f in features]
    return np.concatenate(features, axis=0)

def global_feature_extractor(s, sample_rate=300.0):
    # normalize the signal-window
    s = norm(s)

    # extract 2D features
    s_mfcc = mfcc_2d_features(s, sample_rate)

    return s_mfcc

TRAIN_DIR = 'physionet.org/files/challenge-2017/1.0.0/training/'

SAMPLE_RATE = 300.0
SAVE = './Data/D40/'

WL = 7.0

label_to_class = {'noisy': 0, 'other': 1, 'normal': 2, 'af': 3}
label_to_ws = {'noisy': 0.07, 'other': 0.8, 'normal': 1.7, 'af': 0.25}

annotations = []
labels = []

for cls_ in ['noisy', 'other', 'normal', 'af']:
	file = open('{}RECORDS-{}'.format(TRAIN_DIR, cls_))
	recordings = [f.strip() for f in  file.readlines()]

	annotations += recordings
	labels += [cls_]*len(recordings)

Y = []

INDXT = 0

file_to_sample = {}
file_to_label = {}

for i, (file, label) in enumerate(zip(annotations, labels)):

    if label == 'noisy':
        continue

    mat = scipy.io.loadmat('{}{}.mat'.format(TRAIN_DIR, file))['val']
    mat = np.squeeze(mat, axis=0)
    file_to_label[file] = label

    WS = label_to_ws[label]*(0.3)

    for i in range(0, mat.shape[0], int(WS*SAMPLE_RATE)):
    	if i + int(WL*SAMPLE_RATE) < mat.shape[0]:
            window = mat[i:i+int(SAMPLE_RATE*WL)]
            window = np.nan_to_num(norm(window))
            Y += [label_to_class[label]]

            import neurokit2 as nk
            import torch

            _, r_peaks = nk.ecg_peaks(window, sampling_rate=SAMPLE_RATE)
            r_peaks = r_peaks['ECG_R_Peaks']

            reference = torch.zeros((window.shape[0], ))

            for peak in r_peaks:
                pqrst_ids = torch.arange(peak-20, peak+20, 1)
                pqrst_ids = pqrst_ids[pqrst_ids < window.shape[-1]]

                reference[pqrst_ids] = 1.0

            save_filename = '{}s_{:d}_{:d}.npy'.format(SAVE, label_to_class[label], INDXT)
            np.save(save_filename, window)

            if file in file_to_sample:
                file_to_sample[file].append(save_filename)
            else:
                file_to_sample[file] = [save_filename]

            save_filename = '{}ref_{:d}_{:d}.npy'.format(SAVE, label_to_class[label], INDXT)
            np.save(save_filename, reference)

            save_filename = '{}name_{:d}_{:d}.npy'.format(SAVE, label_to_class[label], INDXT)
            np.save(save_filename, np.array([file]))

            print('----{:d} saving...'.format(INDXT), SAVE, window.shape)
            INDXT += 1

Y = np.array(Y)

print('---',Y.shape)

print(np.where(Y == 1)[0].shape)
print(np.where(Y == 2)[0].shape)
print(np.where(Y == 3)[0].shape)
	
import pickle

with open('{}file_to_sample.pickle'.format(SAVE), 'wb') as handle:
    pickle.dump(file_to_sample, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('{}file_to_label.pickle'.format(SAVE), 'wb') as handle:
    pickle.dump(file_to_label, handle, protocol=pickle.HIGHEST_PROTOCOL)
