import numpy as np
import glob 
from scipy.io import loadmat
import matplotlib.pyplot as plt
import pandas as pd

info = pd.read_csv('metainfo.csv')
code_to_class = {}

for code, cl in zip(info['SNOMED CT Code'], info['Abbreviation']):
    if code not in code_to_class:
        code_to_class[code] = cl

valid = ['IAVB', 'AF', 'AFL', 'Brady', 'CRBBB', 'IRBBB', 'LAnFB', 'LAD'
,'LBBB','LQRSV', 'NSIVCB', 'PR', 'PAC', 'PVC', 'LPR', 'LQT', 'QAb','RAD'
,'RBBB','SA','SB','NSR','STach','SVPB','TAb','TInv','VPB'
]

path = './physionet.org/files/challenge-2020/1.0.2/training/georgia/*'
time = []

def read_annotations(filename):
    fp = open(filename)
    content = fp.readlines()
    fp.close()

    filename, leads, fs, N = content[0].strip().split(' ')

    pathology = None
    for line in content:
        if '# Dx:' in line:
            pathology = line.strip().split(' ')[-1]

    if pathology is None:
        raise Exception

    if ',' in pathology:
        pathology = pathology.split(',')[0] 

    return {'filename': filename, 'fs': int(fs), 'N': int(N), 'pathology': code_to_class[int(pathology)]}

WL = 7
WS = 0.09
N = 0
SAVE = './data/'

classes = {}
data_map = {}

# find classes
class_to_sample = {}
for file in glob.glob(path):
    print(file)
    for record in glob.glob('{}/*.mat'.format(file)):
        ecg = loadmat(record)['val']
        annotations = read_annotations(record.replace('mat', 'hea'))
        class_ = annotations['pathology']

        if annotations['pathology'] not in valid:
            continue

        if class_ in class_to_sample:
            class_to_sample[class_] += 1
        else:
            class_to_sample[class_] = 0

print(class_to_sample)

for file in glob.glob(path):
    print(file)
    for record in glob.glob('{}/*.mat'.format(file)):
        ecg = loadmat(record)['val']
        annotations = read_annotations(record.replace('mat', 'hea'))

        if annotations['pathology'] not in valid:
            print('######## {} not in valid, ignored!'.format(annotations['pathology']))
            continue

        for idx in range(0, annotations['N'], int(WS*annotations['fs'])):
            if idx + WL*annotations['fs'] < ecg.shape[-1]:
                segment = ecg[:, idx:idx + WL*annotations['fs']]
                raw_recording = record.replace('.mat', '').split('/')[-1]
                
                savefilename = '{}ecg_{:d}_{}'.format(SAVE, N, annotations['pathology'])
                N+=1

                if raw_recording in data_map:
                    data_map[raw_recording].append(savefilename)
                else:
                    data_map[raw_recording] = [savefilename]

                np.save(savefilename, segment)

                if segment.shape[-1] != int(3500):
                    print('invalid lenght: ', WL*annotations['fs'], segment.shape[-1])
                    raise Exception

                print('###### saved! - ', savefilename, segment.shape)

import pickle

with open('data_map.pkl', 'wb') as handle:
    pickle.dump(data_map, handle, protocol=pickle.HIGHEST_PROTOCOL)

print(N)
