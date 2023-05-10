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
import glob
from numpy import genfromtxt
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def norm(a):
    return (a - np.min(a)) / np.ptp(a)

def resample_normalize(signal, N):
	ecg_12led = []
	for channel in signal:
		# normalize the signal
		ch_nor = norm(channel)
		# resample
		ch =resample(ch_nor, N)
		# save
		ecg_12led.append(ch)

	ecg_12led = [np.expand_dims(f, axis=0) for f in ecg_12led]
	return np.concatenate(ecg_12led, axis=0)


TRAIN_DIR = './training_set/'

SAMPLE_RATE = 500.0
SAVE = './Data/D13/'

WL = 12.0

files = []
labels = []

annotations = './training_set/REFERENCE.csv'
fp = open(annotations, 'r')
for line in fp.readlines()[1:]:
	file_id, label = line.strip().split(',')[:2]

	files.append(file_id)
	labels.append(int(label)-1)

	if int(label) == 0:
		break

Y = []

label_to_ws = {
	0: 0.15,
	1: 0.2,
	2: 0.1,
	3: 0.05,
	4: 0.4,
	5: 0.2,
	6: 0.2,
	7: 0.2,
	8: 0.05,
}


INDEX_N = 0

for i, (file, label) in enumerate(zip(files, labels)):
	print(file)
	mat = scipy.io.loadmat('{}{}.mat'.format(TRAIN_DIR, file))['ECG'][0][0][2]
	WS = label_to_ws[label]*0.5

	if mat.shape[-1]/SAMPLE_RATE < WL:		
		pad_diff = int((WL*SAMPLE_RATE - mat.shape[-1]))
		pad_mat = np.zeros((12, pad_diff))
		mat = np.concatenate([pad_mat, mat, pad_mat], axis=-1)
		

	for i in range(0, mat.shape[-1], int(WS*SAMPLE_RATE)):
		if i + int(WL*SAMPLE_RATE) < mat.shape[-1]:
			window = mat[:, i:i+int(SAMPLE_RATE*WL)]
			Y += [label]

			window_re = np.nan_to_num(resample_normalize(window, 2000))
		
			#file_name = '{}s_{:d}_{}_{:d}.npy'.format(SAVE, label, file, INDEX_N)
			#np.save(file_name, window)

			file_name = '{}snr_{:d}_{}_{:d}.npy'.format(SAVE, label, file, INDEX_N)
			np.save(file_name, window_re)
			
			print('saving-{:d}-{}'.format(INDEX_N, file), window_re.shape)

			INDEX_N += 1

Y = np.array(Y)
print(Y.shape)
for i in range(9):
	ids = np.where(Y == i)[0]
	print(i,ids.shape[0])


