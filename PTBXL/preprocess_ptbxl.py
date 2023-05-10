import pandas as pd 
import numpy as np
import wfdb 
import ast
import matplotlib.pyplot as plt

meta = pd.read_csv('labels.csv')

def to_list(s):
	s = s.replace('[', '')
	s = s.replace(']', '')
	s = s.replace('\'', '')
	return s.split(',')

def norm(a):
    return (a - np.min(a)) / np.ptp(a)

# extract
WL = 7.0

DIR = 'SAVE_DIR'

label_to_id = {'NORM':0, 'MI':1, 'STTC':2, 'CD':3, 'HYP':4}
label_to_ws = {'NORM':0.3, 'MI':0.08, 'STTC':0.08, 'CD':0.05, 'HYP':0.02}

Y = []
subject_to_recording = {}
R_INDEX = 0

for ecg_id, patient_id, filename_hr, d_classes in zip(meta['ecg_id'], meta['patient_id'], meta['filename_lr'], meta['diagnostic_superclass']):
		d_classes = to_list(d_classes)
		
		if len(d_classes) != 1:
			print(ecg_id,'multi-label-ann:', d_classes)
			continue
		d_class = d_classes[0]

		if d_class == '':
			print(ecg_id, 'no annotation--', d_class)
			continue

		print(ecg_id, 'valid-processing')

		record = wfdb.rdrecord('ptbxl/{}'.format(filename_hr))
		signal = record.p_signal
		fs = record.fs
		WS = 0.5*label_to_ws[d_class]
		
		for idx in range(0, signal.shape[0], int(fs*WS)):
			if idx + int(fs*WL) < signal.shape[0]:
				window = signal[idx:idx + int(WL*fs)]

				Y += [label_to_id[d_class]]
				# save
				save_filename = '{}s_{:d}_{:d}_{:d}.npy'.format(DIR, label_to_id[d_class], int(patient_id), R_INDEX)
				np.save(save_filename, window)
				
				R_INDEX += 1
				if patient_id in subject_to_recording:
					subject_to_recording[patient_id] += [save_filename]
				else:
					subject_to_recording[patient_id] = [save_filename]

				print('saving-file: ', save_filename, window.shape)

import pickle
with open('{}subject_to_recordings.pickle'.format(DIR), 'wb') as handle:
    pickle.dump(subject_to_recording, handle, protocol=pickle.HIGHEST_PROTOCOL)

Y = np.array(Y)
print(Y.shape)
print(np.where(Y == 0)[0].shape)
print(np.where(Y == 1)[0].shape)
print(np.where(Y == 2)[0].shape)
print(np.where(Y == 3)[0].shape)
print(np.where(Y == 4)[0].shape)
