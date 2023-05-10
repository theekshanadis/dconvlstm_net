
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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

        if 'name_' in ID:
            ID = ID.replace('/name_', '/s_', )

        X = torch.from_numpy(np.load(ID)).float()
        y = int(ID.split('_')[1]) - 1  # we removed the noicy

        return X, y


class DataSetTest(torch.utils.data.Dataset):
    def __init__(self, file_names):
        self.file_names = file_names

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        ID = self.file_names[index]
        ID = ID.replace('/name_', '/s_', )

        try:
            X = np.load(ID)
        except Exception as e:
            print(e, ID)

        X = torch.from_numpy(np.load(ID)).float()

        if torch.isnan(X).any():
            print('invalid---')

        y = int(ID.split('_')[1]) - 1

        name = np.load(ID.replace('/s_', '/name_'))[0]
        return X, y, name


class DilatedProp(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_feat_maps=12):
        super(DilatedProp, self).__init__()

        self.dilation_fac = [1, 2, 3, 4]

        self.dil_cnv_block = torch.nn.ModuleList(
            nn.Conv1d(in_channels, num_feat_maps, kernel_size=3, padding=self.p(d, 3), dilation=d) for d in
            self.dilation_fac
        )

        self.lstm_hidden = nn.LSTM(num_feat_maps * len(self.dilation_fac), hidden_size=in_channels, batch_first=True,
                                   num_layers=1)

        self.gate = torch.nn.Linear(2 * in_channels, in_channels)
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

        self.dl1 = DilatedProp(1, 24, num_feat_maps=12)
        self.dl2 = DilatedProp(24, 24, num_feat_maps=12)
        self.dl3 = DilatedProp(24, 32, num_feat_maps=12)
        self.dl4 = DilatedProp(32, 32, num_feat_maps=16)
        self.dl5 = DilatedProp(32, 48, num_feat_maps=16)
        self.dl6 = DilatedProp(48, 48, num_feat_maps=24)
        self.dl7 = DilatedProp(48, 64, num_feat_maps=24)
        self.dl8 = DilatedProp(64, 72, num_feat_maps=24)
        self.dl9 = DilatedProp(72, 84, num_feat_maps=24)

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(252, 100, bias=True),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(100, 40, bias=True),
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

        self.lstm_hidden = nn.LSTM(num_feat_maps * len(self.dilation_fac), hidden_size=in_channels, batch_first=True,
                                   num_layers=1)

        self.gate = torch.nn.Linear(2 * in_channels, in_channels)
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

        self.dl1 = DilatedPropB(1, 24, dilations=[1, 2, 3, 4, 5, 7], num_feat_maps=6)
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

        self.dl1 = DilatedPropB(1, 24, dilations=[1, 2, 3, 4, 5, 7], num_feat_maps=6)
        self.dl2 = DilatedPropB(24, 24, dilations=[1, 2, 3, 4, 5], num_feat_maps=8)
        self.dl3 = DilatedPropB(24, 32, dilations=[1, 2, 3, 4, 5], num_feat_maps=12)
        self.dl4 = DilatedPropB(32, 32, dilations=[1, 2, 3, 4, 5], num_feat_maps=12)
        self.dl5 = DilatedPropB(32, 48, dilations=[1, 2, 3, 4], num_feat_maps=16)
        self.dl6 = DilatedPropB(48, 48, dilations=[1, 2, 3, 4], num_feat_maps=16)
        self.dl7 = DilatedPropB(48, 64, dilations=[1, 2, 3], num_feat_maps=36)
        self.dl8 = DilatedPropB(64, 72, dilations=[1, 2, 3], num_feat_maps=42)
        self.dl9 = DilatedPropB(72, 72, dilations=[1, 2], num_feat_maps=42)
        self.dl10 = DilatedPropB(72, 72, dilations=[1], num_feat_maps=48)

        self.mlp = torch.nn.Sequential(
            torch.nn.Dropout(0.4),
            torch.nn.Linear(216, 40, bias=True),
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

        self.dl1 = DilatedPropB(1, 24, dilations=[1, 2, 3, 4, 5, 7], num_feat_maps=4)
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
            torch.nn.Dropout(0.4),
            torch.nn.Linear(144, 40, bias=True),
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


class DilatedLSTMResF(torch.nn.Module):
    def __init__(self):
        super(DilatedLSTMResF, self).__init__()

        self.dl1 = DilatedPropB(1, 24, dilations=[1, 2], num_feat_maps=12)
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
            torch.nn.Linear(144, 40, bias=True),
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


class SEComponent(torch.nn.Module):
    def __init__(self, n_channels):
        super(SEComponent, self).__init__()
        self.squeeze = torch.nn.AdaptiveAvgPool1d(1)
        self.exitation = torch.nn.Sequential(
            torch.nn.Linear(n_channels, n_channels, bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(n_channels, n_channels // 2, bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(n_channels // 2, n_channels, bias=True),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _ = x.shape

        y = self.squeeze(x).view(bs, c)
        y = self.exitation(y).view(bs, c, 1)

        return x * y.expand_as(x)


class DilatedPropE(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dilations, num_feat_maps=12):
        super(DilatedPropE, self).__init__()

        self.dilation_fac = dilations
        self.dil_cnv_block = torch.nn.ModuleList(
            nn.Conv1d(in_channels, num_feat_maps, kernel_size=3, padding=self.p(d, 3), dilation=d) for d in
            self.dilation_fac
        )
        self.chan_att = SEComponent(num_feat_maps * len(dilations))
        self.cnv1d_transform = nn.Conv1d(num_feat_maps * len(self.dilation_fac) + in_channels, out_channels,
                                         kernel_size=3, dilation=1, padding=1)
        self.lstm_hidden = nn.LSTM(out_channels, hidden_size=out_channels, batch_first=True, num_layers=1)

        self.li_norm = nn.InstanceNorm1d(in_channels)
        self.lo_norm = nn.InstanceNorm1d(out_channels)

    def p(self, d, k):
        return int((d * (k - 1)) / 2)

    def forward(self, x):
        x = self.li_norm(x)
        residual = x
        # dilated propagation
        x_dil = []
        for layer in self.dil_cnv_block:
            x_dil.append(layer(x))
        x_dil = torch.cat(x_dil, dim=1)
        x_dil = F.relu(x_dil)
        x_dil = F.dropout(x_dil, 0.4)
        # channel-attention
        x_dil = self.chan_att(x_dil)

        # convolution-transformation
        x = torch.cat([residual, x_dil], dim=1)
        x = self.cnv1d_transform(x).relu()
        x = F.dropout(x, 0.4)

        # sequential-processing
        x, _ = self.lstm_hidden(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        x = F.dropout(x, 0.4)
        x = self.lo_norm(x)

        return x


class DilatedLSTMResE(torch.nn.Module):
    def __init__(self):
        super(DilatedLSTMResE, self).__init__()

        self.dl1 = DilatedPropE(1, 24, dilations=[1, 2, 3, 4, 5, 7], num_feat_maps=4)
        self.dl2 = DilatedPropE(24, 24, dilations=[1, 2, 3, 4, 5], num_feat_maps=8)
        self.dl3 = DilatedPropE(24, 32, dilations=[1, 2, 3, 4, 5], num_feat_maps=12)
        self.dl5 = DilatedPropE(32, 36, dilations=[1, 2, 3, 4], num_feat_maps=10)
        self.dl6 = DilatedPropE(36, 48, dilations=[1, 2, 3, 4], num_feat_maps=12)
        self.dl7 = DilatedPropE(48, 56, dilations=[1, 2, 3], num_feat_maps=20)
        self.dl8 = DilatedPropE(56, 64, dilations=[1, 2, 3], num_feat_maps=22)
        self.dl9 = DilatedPropE(64, 72, dilations=[1, 2], num_feat_maps=38)
        self.dl10 = DilatedPropE(72, 72, dilations=[1], num_feat_maps=72)

        self.mlp = torch.nn.Sequential(
            torch.nn.Dropout(0.4),
            torch.nn.Linear(144, 40, bias=True),
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
        x = F.avg_pool1d(x, kernel_size=4)

        x = self.dl5(x)
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

model = DilatedLSTMRes()  # modal we tested, we have also given some other implementations as well. 
model.cuda()

optim = torch.optim.Adam(model.parameters(), lr=0.0001)
loss_function = torch.nn.CrossEntropyLoss()

fold = 4
code = 'DilatedLSTMRes-12s-phy3_noise-f{:d}'.format(fold)
model.cuda()

# load the saved baselines
try:
    model.load_state_dict(torch.load('./Models/{}.tm'.format(code)))
    optim.load_state_dict('./Models/{}_opt.tm'.format(code))
    print('loaded the saved modules')
except Exception as e:
    print(e)

print('# of parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))

# subject-independent evaluations
import pickle

recording_to_files = None
with open(r"file_to_sample.pickle", "rb") as input_file:
    recording_to_files = pickle.load(input_file)
files_train = np.load('./Folds/train_{:d}_12s.npy'.format(fold))
files_test = np.load('./Folds/test_{:d}_12s.npy'.format(fold))

X_train = []
for file in files_train:
    if file in recording_to_files:
        X_train += recording_to_files[file]

X_test = []
for file in files_test:
    if file in recording_to_files:
        X_test += recording_to_files[file]

def remove_noicy(X):
    filtered = []
    for x in X:
        y = int(x.split('_')[1])
        if y != 0:
            filtered.append(x)
    return filtered

# load the augmented data (if available)
file_to_samples_aug = None
with open(r"./Data/D12/file_to_samples_aug.pickle", "rb") as input_file:
    file_to_samples_aug = pickle.load(input_file)
# load the augmented data-2
file_to_samples_aug_re = None
with open(r"./Data/D12/file_to_samples_aug_re.pickle", "rb") as input_file:
    file_to_samples_aug_re = pickle.load(input_file)

X_aug = []
for file in files_train:
    if file in file_to_samples_aug:
        X_aug += file_to_samples_aug[file]
        X_aug += file_to_samples_aug_re[file]

X_train = np.array(remove_noicy(X_train + X_aug))
X_test = np.array(remove_noicy(X_test))

print('train on: {:d}, validate on {:d} augmented {:d}'.format(X_train.shape[0], X_test.shape[0], len(X_aug)))

params = {'batch_size': 50,
          'shuffle': True,
          'num_workers': 4}

training_set = DataSet(X_train)
training_generator = torch.utils.data.DataLoader(training_set, **params)

validation_set = DataSet(X_test)
validation_generator = torch.utils.data.DataLoader(validation_set, **params)

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

        print('\r[{:4d}]train loss: {:.4f} accuracy {:.4f} f1 {:.4f}'.format(i, loss.item(), acc, f1), end='')

        test_loss.append(loss.item())
        test_acc.append(acc)
        test_f1.append(f1)

    print('\n------({}) Test Loss: {:.4f} accuracy {:.4f} f1 {:.4f}, maxF1: {:.4f}'.format(code, np.mean(test_loss),
                                                                                           np.mean(test_acc),
                                                                                           np.mean(test_f1), val_f1))

    if np.mean(test_f1) > val_f1:
        val_f1 = np.mean(test_f1)
        torch.save(model.state_dict(), './Models/{}.tm'.format(code))
        torch.save(optim, './Models/{}_opt.tm'.format(code))

'''
############################ --- used for signal-level validation
print('testing the model: ', code)

validation_set = DataSetTest(X_test)
validation_generator = torch.utils.data.DataLoader(validation_set, **params)

model.eval()
Y, F1, Y_tr, y_prob = [], [], [], []
for i, (mfcc, y, name) in enumerate(validation_generator):
    y_hat = model(mfcc.cuda())
    loss = loss_function(y_hat, y.cuda())

    acc = accuracy_score(y_hat.argmax(dim=-1).detach().cpu().numpy(), y)
    f1 = f1_score(y_hat.argmax(dim=-1).detach().cpu().numpy(), y, average='micro')

    Y += y_hat.argmax(dim=-1).detach().cpu().numpy().tolist()
    F1 += list(name)
    Y_tr += y.detach().tolist()
    y_prob += y_hat.detach().tolist()

    print('\r[{:4d}]test loss: {:.4f} accuracy {:.4f} f1 {:.4f}'.format(i, loss.item(), acc, f1), end='')


def mode(a):
    counts = np.bincount(a)
    return np.argmax(counts)


Y = np.array(Y)
Y_tr = np.array(Y_tr)
F1 = np.array(F1)
y_prob = np.array(y_prob)

print(F1.shape, Y.shape, y_prob.shape)
np.save('y_true_af.npy', Y)
np.save('y_pred_af.npy', y_prob)

exit()

f_ids = np.unique(F1)

y_true, y_pred = [], []

for f in f_ids:
    ids = np.where(F1 == f)
    y_true.append(Y_tr[ids][0])
    y_pred.append(mode(Y[ids]))

print(accuracy_score(y_true, y_pred))
print(f1_score(y_true, y_pred, average='micro'))
print('f1:-marco', f1_score(y_true, y_pred, average='macro'))

print(confusion_matrix(y_true, y_pred))

print(classification_report(y_true, y_pred, target_names=['Other', 'Normal', 'AF']))

conf = confusion_matrix(y_true, y_pred)

df_cm = pd.DataFrame(conf, index=[i for i in ['Other', 'Normal', 'AF']],
                     columns=[i for i in ['Other', 'Normal', 'AF']])

plt.figure(figsize=(10, 7))
sn.heatmap(df_cm, annot=True)
plt.savefig('conf-{}.png'.format(code))
exit()
'''
