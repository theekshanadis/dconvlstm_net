import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
from sklearn.metrics import plot_confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

class DataSet(torch.utils.data.Dataset):
    def __init__(self, file_names):
        self.file_names = file_names

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        ID = self.file_names[index]

        try:
            X = np.load(ID)
        except Exception as e:
            print(e, ID)
    
        X = torch.from_numpy(np.load(ID)).float()

        if torch.isnan(X).any():
            print('invalid---')

        y = int(ID.split('_')[1])

        return X, y

class DataSetTest(torch.utils.data.Dataset):
    def __init__(self, file_names):
        self.file_names = file_names

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        ID = self.file_names[index]

        try:
            X = np.load(ID)
        except Exception as e:
            print(e, ID)
    
        X = torch.from_numpy(np.load(ID))

        if torch.isnan(X).any():
            print('invalid---')

        y = int(ID.split('_')[1])

        name = ID.split('_')[2]

        return X, y, name

class DilatedProp(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_feat_maps=12):
        super(DilatedProp, self).__init__()

        self.dilation_fac = [1, 2, 3, 4]

        self.dil_cnv_block = torch.nn.ModuleList(
            nn.Conv1d(in_channels, num_feat_maps, kernel_size=3, padding=self.p(d, 3), dilation=d) for d in
            self.dilation_fac
        )

        self.lstm_hidden = nn.LSTM(num_feat_maps*len(self.dilation_fac), hidden_size=in_channels, batch_first=True, num_layers=1)

        self.gate = torch.nn.Linear(2*in_channels, in_channels)
        self.cnv1d_transform = nn.Conv1d(in_channels, out_channels, kernel_size=3, dilation=1, padding=1)

        self.bn_input = nn.BatchNorm1d(in_channels)
        self.bn_output = nn.BatchNorm1d(out_channels)

    def p(self, d, k):
        return int((d * (k - 1)) / 2)

    def forward(self, x):
        x = self.bn_input(x)
        residual = x
        # dilated propagation
        x_dil = []
        for layer in self.dil_cnv_block:
            x_dil.append(layer(x))
        x_dil = torch.cat(x_dil, dim=1)

        x_dil = F.relu(x_dil)
        x_dil = F.dropout(x_dil, 0.4)

        x_dil, _ = self.lstm_hidden(x_dil.permute(0, 2, 1))
        x_dil = x_dil.permute(0, 2, 1)
        x_dil = F.dropout(x_dil, 0.4)

        x = torch.cat([residual, x_dil], dim=1).permute(0, 2, 1)
        x = self.gate(x).relu()
        x = x.permute(0, 2, 1)

        x = self.cnv1d_transform(x).relu()
        x = F.dropout(x, 0.4)
        x = self.bn_output(x)

        return x

class DilatedLSTMRes(torch.nn.Module):
    def __init__(self):
        super(DilatedLSTMRes, self).__init__()

        self.dl1 = DilatedProp(12, 24, num_feat_maps=12)
        self.dl2 = DilatedProp(24, 24, num_feat_maps=12)
        self.dl3 = DilatedProp(24, 32, num_feat_maps=12)
        self.dl4 = DilatedProp(32, 32, num_feat_maps=16)
        self.dl5 = DilatedProp(32, 48, num_feat_maps=16)
        self.dl6 = DilatedProp(48, 48, num_feat_maps=24)
        self.dl7 = DilatedProp(48, 64, num_feat_maps=24)
        self.dl8 = DilatedProp(64, 72, num_feat_maps=24)
        self.dl9 = DilatedProp(72, 84, num_feat_maps=24)

        self.mlp = torch.nn.Sequential(
            torch.nn.Dropout(0.4),
            torch.nn.Linear(168, 9, bias=True),
            torch.nn.Softmax(dim=-1)
        )

    def forward(self, x):
        if len(x.shape) != 3:
            x = x.unsqueeze(dim=1)

        x = self.dl1(x)
        x = self.dl2(x)
        x = F.avg_pool1d(x, kernel_size=4)

        x = self.dl3(x)
        x = self.dl4(x)
        x = F.avg_pool1d(x, kernel_size=3)

        x = self.dl5(x)
        x = F.avg_pool1d(x, kernel_size=3)
        x = self.dl6(x)
        x = F.avg_pool1d(x, kernel_size=3)

        x = self.dl7(x)
        x = self.dl8(x)
        x = F.avg_pool1d(x, kernel_size=3)

        x = self.dl9(x)
        x = F.avg_pool1d(x, kernel_size=3).flatten(start_dim=1)
        return self.mlp(x)

class DilatedPropB(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dilations, num_feat_maps=12):
        super(DilatedPropB, self).__init__()

        self.dilation_fac = dilations

        self.dil_cnv_block = torch.nn.ModuleList(
            nn.Conv1d(in_channels, num_feat_maps, kernel_size=3, padding=self.p(d, 3), dilation=d) for d in
            self.dilation_fac
        )

        self.lstm_hidden = nn.LSTM(num_feat_maps*len(self.dilation_fac), hidden_size=in_channels, batch_first=True, num_layers=1)

        self.gate = torch.nn.Linear(2*in_channels, in_channels)
        self.cnv1d_transform = nn.Conv1d(in_channels, out_channels, kernel_size=3, dilation=1, padding=1)

        self.bn_input = nn.BatchNorm1d(in_channels)
        self.bn_output = nn.BatchNorm1d(out_channels)

    def p(self, d, k):
        return int((d * (k - 1)) / 2)

    def forward(self, x):
        x = self.bn_input(x)
        residual = x
        # dilated propagation
        x_dil = []
        for layer in self.dil_cnv_block:
            x_dil.append(layer(x))
        x_dil = torch.cat(x_dil, dim=1)

        x_dil = F.relu(x_dil)
        x_dil = F.dropout(x_dil, 0.4)

        x_dil, _ = self.lstm_hidden(x_dil.permute(0, 2, 1))
        x_dil = x_dil.permute(0, 2, 1)
        x_dil = F.dropout(x_dil, 0.4)

        x = torch.cat([residual, x_dil], dim=1).permute(0, 2, 1)
        x = self.gate(x).relu()
        x = x.permute(0, 2, 1)

        x = self.cnv1d_transform(x).relu()
        x = F.dropout(x, 0.4)
        x = self.bn_output(x)

        return x

class DilatedLSTMResB(torch.nn.Module):
    def __init__(self):
        super(DilatedLSTMResB, self).__init__()

        self.dl1 = DilatedPropB(12, 24, dilations=[1, 2, 3, 4, 5, 7], num_feat_maps=6)
        self.dl2 = DilatedPropB(24, 24, dilations=[1, 2, 3, 4], num_feat_maps=8)
        self.dl3 = DilatedPropB(24, 32, dilations=[1, 2, 3, 4], num_feat_maps=12)
        self.dl4 = DilatedPropB(32, 32, dilations=[1, 2, 3, 4], num_feat_maps=12)
        self.dl5 = DilatedPropB(32, 48, dilations=[1, 2, 3], num_feat_maps=16)
        self.dl6 = DilatedPropB(48, 48, dilations=[1, 2, 3], num_feat_maps=22)
        self.dl7 = DilatedPropB(48, 64, dilations=[1, 2, 3], num_feat_maps=36)
        self.dl8 = DilatedPropB(64, 72, dilations=[1, 2, 3], num_feat_maps=32)
        self.dl9 = DilatedPropB(72, 84, dilations=[1, 2, 3], num_feat_maps=32)
        self.dl10 = DilatedPropB(84, 96, dilations=[1, 2, 3], num_feat_maps=36)

        self.mlp = torch.nn.Sequential(
            torch.nn.Dropout(0.4),
            torch.nn.Linear(96, 40, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(40, 3, bias=True),
            torch.nn.Softmax(dim=-1)
        )

    def forward(self, x):
        if len(x.shape) != 3:
            x = x.unsqueeze(dim=1)

        x = self.dl1(x)
        x = self.dl2(x)
        x = F.avg_pool1d(x, kernel_size=4)

        x = self.dl3(x)
        x = self.dl4(x)
        x = F.avg_pool1d(x, kernel_size=3)

        x = self.dl5(x)
        x = F.avg_pool1d(x, kernel_size=3)
        x = self.dl6(x)
        x = F.avg_pool1d(x, kernel_size=3)

        x = self.dl7(x)
        x = self.dl8(x)
        x = F.avg_pool1d(x, kernel_size=3)

        x = self.dl9(x)
        x = F.avg_pool1d(x, kernel_size=3)

        x = self.dl10(x)
        x = F.avg_pool1d(x, kernel_size=3).flatten(start_dim=1)

        return self.mlp(x)


class DilatedLSTMResC(torch.nn.Module):
    def __init__(self):
        super(DilatedLSTMResC, self).__init__()

        self.dl1 = DilatedPropB(12, 24, dilations=[1, 2, 3, 4, 5, 7], num_feat_maps=4)
        self.dl2 = DilatedPropB(24, 24, dilations=[1, 2, 3, 4, 5], num_feat_maps=8)
        self.dl3 = DilatedPropB(24, 32, dilations=[1, 2, 3, 4, 5], num_feat_maps=12)
        self.dl4 = DilatedPropB(32, 32, dilations=[1, 2, 3, 4, 5], num_feat_maps=12)
        self.dl5 = DilatedPropB(32, 48, dilations=[1, 2, 3, 4], num_feat_maps=16)
        self.dl6 = DilatedPropB(48, 48, dilations=[1, 2, 3, 4], num_feat_maps=16)
        self.dl7 = DilatedPropB(48, 64, dilations=[1, 2, 3], num_feat_maps=36)
        self.dl8 = DilatedPropB(64, 72, dilations=[1, 2, 3], num_feat_maps=36)
        self.dl9 = DilatedPropB(72, 72, dilations=[1, 2], num_feat_maps=38)
        self.dl10 = DilatedPropB(72, 72, dilations=[1, 2], num_feat_maps=36)
        self.dl11 = DilatedPropB(72, 72, dilations=[1], num_feat_maps=72)

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(72, 9, bias=True),
            torch.nn.Softmax(dim=-1)
        )

    def forward(self, x):
        if len(x.shape) != 3:
            x = x.unsqueeze(dim=1)

        x = self.dl1(x)
        x = self.dl2(x)
        x = F.avg_pool1d(x, kernel_size=4)

        x = self.dl3(x)
        x = self.dl4(x)
        x = F.avg_pool1d(x, kernel_size=2)

        x = self.dl5(x)
        x = self.dl6(x)
        x = F.avg_pool1d(x, kernel_size=3)

        x = self.dl7(x)
        x = self.dl8(x)
        x = F.avg_pool1d(x, kernel_size=3)

        x = self.dl9(x)
        x = F.avg_pool1d(x, kernel_size=3)

        x = self.dl10(x)
        x = F.avg_pool1d(x, kernel_size=3)

        x = self.dl11(x)
        x = F.avg_pool1d(x, kernel_size=3).flatten(start_dim=1)
        
        return self.mlp(x)


class DilatedLSTMResF(torch.nn.Module):
    def __init__(self):
        super(DilatedLSTMResF, self).__init__()

        self.dl1 = DilatedPropB(12, 24, dilations=[1, 2], num_feat_maps=12)
        self.dl2 = DilatedPropB(24, 24, dilations=[1, 2], num_feat_maps=12)
        self.dl3 = DilatedPropB(24, 32, dilations=[1, 2], num_feat_maps=16)
        self.dl4 = DilatedPropB(32, 32, dilations=[1, 2], num_feat_maps=16)
        self.dl5 = DilatedPropB(32, 48, dilations=[1, 2], num_feat_maps=24)
        self.dl6 = DilatedPropB(48, 48, dilations=[1, 2], num_feat_maps=24)
        self.dl7 = DilatedPropB(48, 64, dilations=[1, 2], num_feat_maps=36)
        self.dl8 = DilatedPropB(64, 72, dilations=[1, 2], num_feat_maps=36)
        self.dl9 = DilatedPropB(72, 72, dilations=[1, 2], num_feat_maps=38)
        self.dl10 = DilatedPropB(72, 72, dilations=[1, 2], num_feat_maps=48)
        self.dl11 = DilatedPropB(72, 72, dilations=[1, 2], num_feat_maps=36)

        self.mlp = torch.nn.Sequential(
            torch.nn.Dropout(0.4),
            torch.nn.Linear(72, 40, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(40, 9, bias=True),
            torch.nn.Softmax(dim=-1)
        )

    def forward(self, x):
        if len(x.shape) != 3:
            x = x.unsqueeze(dim=1)

        x = self.dl1(x)
        x = self.dl2(x)
        x = F.avg_pool1d(x, kernel_size=4)

        x = self.dl3(x)
        x = self.dl4(x)
        x = F.avg_pool1d(x, kernel_size=2)

        x = self.dl5(x)
        x = self.dl6(x)
        x = F.avg_pool1d(x, kernel_size=3)

        x = self.dl7(x)
        x = self.dl8(x)
        x = F.avg_pool1d(x, kernel_size=2)

        x = self.dl9(x)
        x = F.avg_pool1d(x, kernel_size=3)

        x = self.dl10(x)
        x = F.avg_pool1d(x, kernel_size=3)

        x = self.dl11(x)
        x = F.avg_pool1d(x, kernel_size=3).flatten(start_dim=1)
        
        return self.mlp(x)

class BackBone(torch.nn.Module):
    def __init__(self, in_channels):
        super(BackBone, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channels, 64, kernel_size=3, dilation=1)
        self.conv2 = torch.nn.Conv1d(64, 64, kernel_size=3, dilation=1)

        self.conv3 = torch.nn.Conv1d(64, 128, kernel_size=3, dilation=1)
        self.conv4 = torch.nn.Conv1d(128, 128, kernel_size=3, dilation=1)

        self.conv5 = torch.nn.Conv1d(128, 256, kernel_size=3, dilation=1)
        self.conv6 = torch.nn.Conv1d(256, 256, kernel_size=3, dilation=1)
        self.conv7 = torch.nn.Conv1d(256, 256, kernel_size=3, dilation=1)

    def forward(self, x):
        x = self.conv1(x).relu()
        x = self.conv2(x).relu()
        x = F.max_pool1d(x, 3, stride=3)

        x = self.conv3(x).relu()
        x = self.conv4(x).relu()
        x = F.max_pool1d(x, 3, stride=3)

        x = self.conv5(x).relu()
        x = self.conv6(x).relu()
        x = self.conv7(x).relu()
        x = F.max_pool1d(x, 3, stride=3)

        return x


class Branches(torch.nn.Module):
    def __init__(self, p, dilation=1):
        super(Branches, self).__init__()
        p = self.p(dilation, 3)
        self.conv1 = torch.nn.Conv1d(256, 512, kernel_size=3, dilation=dilation, padding=p)
        self.conv2 = torch.nn.Conv1d(512, 512, kernel_size=3, dilation=dilation, padding=p)
        self.conv3 = torch.nn.Conv1d(512, 512, kernel_size=3, dilation=dilation, padding=p)

        self.conv4 = torch.nn.Conv1d(512, 512, kernel_size=3, dilation=dilation, padding=p)
        self.conv5 = torch.nn.Conv1d(512, 256, kernel_size=3, dilation=dilation, padding=p)
        self.conv6 = torch.nn.Conv1d(256, 128, kernel_size=3, dilation=dilation, padding=p)

    def p(self, d, k):
        return int((d * (k - 1)) / 2)

    def forward(self, x):
        x = self.conv1(x).relu()
        x = self.conv2(x).relu()
        x = self.conv3(x).relu()
        x = F.max_pool1d(x, 3, stride=3)

        x = self.conv4(x).relu()
        x = self.conv5(x).relu()
        x = self.conv6(x).relu()

        return x


class Baseline(torch.nn.Module):
    def __init__(self):
        super(Baseline, self).__init__()
        self.backbone = BackBone(12)
        self.branch_1 = Branches(dilation=1, p=0)
        self.branch_2 = Branches(dilation=3, p=1)

        self.linear_fused = torch.nn.Linear(256, 9, bias=True)

        self.lin_1 = nn.Linear(128, 9)
        self.lin_2 = nn.Linear(128, 9)

        self.avg_pool = torch.nn.AvgPool1d(kernel_size=(7))

    def forward(self, x):
        if len(x.shape) != 3:
            x = x.unsqueeze(dim=1)

        x = self.backbone(x)

        x1 = self.branch_1(x)
        x2 = self.branch_2(x)

        x = torch.cat([x1, x2], dim=1)

        x1_out = self.lin_1(F.adaptive_avg_pool1d(x1, 1).squeeze(dim=-1))
        x2_out = self.lin_2(F.adaptive_avg_pool1d(x2, 1).squeeze(dim=-1))

        x = F.adaptive_avg_pool1d(x, 1).squeeze(dim=-1)

        return self.linear_fused(x).softmax(dim=-1), x1_out.softmax(dim=-1), x2_out.softmax(dim=-1)

model = DilatedLSTMResC()
optim = torch.optim.Adam(model.parameters(), lr=0.0001)
loss_function =  torch.nn.CrossEntropyLoss()
fold = 1
#code = 'CNNModelSECH-12s-phy3-f{:d}'.format(fold)
code = 'DilatedLSTMResC-12s-cpsc-au-f{:d}'.format(fold)
model.cuda()

try:
    model.load_state_dict(torch.load('./Models/{}.tm'.format(code)))
    optm = torch.load('./Models/{}_opt.tm'.format(code))
    print('loaded the saved modules')
except Exception as e:
    print(e)

print('# of parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))


import glob
X_train = np.array(glob.glob('./Data/D9/snr_*'))

# file to samples map
file_to_samples = {}

for file in X_train:
    recording = file.split('_')[2]
    if recording in file_to_samples:
        file_to_samples[recording] += [file]
    else:
        file_to_samples[recording] = [file]

files_train = np.load('./Folds/train_d9_{:d}.npy'.format(fold))
files_test = np.load('./Folds/test_d9_{:d}.npy'.format(fold))

X_train = []
for file in files_train:
    if file in file_to_samples:
        X_train += file_to_samples[file]

X_test = []
for file in files_test:
    if file in file_to_samples:
        X_test += file_to_samples[file]

import _pickle as cPickle
# augmented data
X_aug = []
aut_1 = None
with open(r"./Data/D14/file_to_samples_aug_re1.pickle", "rb") as input_file:
    aut_1 = cPickle.load(input_file)

aut_2 = None
with open(r"./Data/D14/file_to_samples_aug_re2.pickle", "rb") as input_file:
    aut_2 = cPickle.load(input_file)

for file in files_train:
    if file in aut_2 and file in aut_1:
        X_aug += aut_1[file] + aut_2[file]

X_train = np.array(X_train + X_aug)
X_test = np.array(X_test)

print('train on: {:d}, validate on {:d}'.format(X_train.shape[0], X_test.shape[0]), len(X_aug))

params = {'batch_size': 100,
          'shuffle': True,
          'num_workers': 4}

training_set = DataSet(X_train)
training_generator = torch.utils.data.DataLoader(training_set, **params)

validation_set = DataSet(X_test)
validation_generator = torch.utils.data.DataLoader(validation_set, **params)

validation_set = DataSetTest(X_test)
validation_generator = torch.utils.data.DataLoader(validation_set, **params)

file_to_labels = {}
annotations = './training_set/REFERENCE.csv'
fp = open(annotations, 'r')
for line in fp.readlines()[1:]:
    file_id, label, label1, label2 = line.strip().split(',')

    file_to_labels[file_id] = [int(label)-1]

    if label1 != '':
        file_to_labels[file_id].append(int(label1)-1)

    if label2 != '':
        file_to_labels[file_id].append(int(label2)-1)

val_f1 = -np.inf

for i in range(400):

    print('---Epoch---{:d}-{}'.format(i, code))

    model.train()
    train_loss, train_acc = [], []
    for i, (mfcc, y) in enumerate(training_generator):
        optim.zero_grad()
        y_hat = model(mfcc.cuda())
        
        loss = loss_function(y_hat, y.cuda())
        loss.backward()
        optim.step()

        acc = accuracy_score(y_hat.argmax(dim=-1).detach().cpu().numpy(), y)
        f1 = f1_score(y_hat.argmax(dim=-1).detach().cpu().numpy(), y, average='micro')
        print('\r[{:4d}]train loss: {:.4f} accuracy {:.4f} f1 {:.4f}'.format(i, loss.item(), acc, f1), end='')
        
        train_loss.append(loss.item())
        train_acc.append(acc)

    print('\n---({}) Train Loss: {:.4f} accuracy {:.4f}'.format(code, np.mean(train_loss), np.mean(train_acc)))


    model.eval()
    test_loss, test_acc, test_f1 = [], [], []
    for i, (mfcc, y) in enumerate(validation_generator):
        y_hat = model(mfcc.cuda())
        loss = loss_function(y_hat, y.cuda())

        acc = accuracy_score(y_hat.argmax(dim=-1).detach().cpu().numpy(), y)
        f1 = f1_score(y_hat.argmax(dim=-1).detach().cpu().numpy(), y, average='micro')

        print('\r[{:4d}]test loss: {:.4f} accuracy {:.4f} f1 {:.4f}'.format(i, loss.item(), acc, f1), end='')
        
        test_loss.append(loss.item())
        test_acc.append(acc)
        test_f1.append(f1)

    print('\n------({}) Test Loss: {:.4f} accuracy {:.4f} f1 {:.4f}, maxF1: {:.4f}'.format(code, np.mean(test_loss), np.mean(test_acc), np.mean(test_f1), val_f1))


    if np.mean(test_f1) > val_f1:
        val_f1 = np.mean(test_f1)
        torch.save(model.state_dict(), './Models/{}.tm'.format(code))
        torch.save(optim, './Models/{}_opt.tm'.format(code))

