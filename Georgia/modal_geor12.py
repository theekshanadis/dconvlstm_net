import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import init
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import KFold
import torchaudio
from sklearn.metrics import f1_score, confusion_matrix

valid = ['IAVB', 'AF', 'AFL', 'Brady', 'CRBBB', 'IRBBB', 'LAnFB', 'LAD'
,'LBBB','LQRSV', 'NSIVCB', 'PR', 'PAC', 'PVC', 'LPR', 'LQT', 'QAb','RAD'
,'RBBB','SA','SB','NSR','STach','SVPB','TAb','TInv','VPB'
]

class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, segments):
        self.segments = segments
        self.resample = torchaudio.transforms.Resample(500, 250)
        
  def __len__(self):
        return len(self.segments)

  def norm(self, x):
    return (x - np.min(x))/np.ptp(x)

  def __getitem__(self, index):
        filename = self.segments[index]
        y = valid.index(filename.split('_')[-1])
        
        try:
            segment = torch.from_numpy(self.norm(np.load(filename + '.npy'))).float()
        except Exception:
            print(filename)
            exit()
    
        segment = self.resample(segment)

        return segment, y

class myConv(nn.Module):
    def __init__(self, in_size, out_size,filter_size=3,stride=1,pad=None,do_batch=1,dov=0):
        super().__init__()
        
        pad=int((filter_size-1)/2)
        
        self.do_batch=do_batch
        self.dov=dov
        self.conv=nn.Conv1d(in_size, out_size,filter_size,stride,pad)
        self.bn=nn.BatchNorm1d(out_size,momentum=0.1)
        
        if self.dov>0:
            self.do=nn.Dropout(dov)
            
    def swish(self,x):
        return x * F.sigmoid(x)
    
    def forward(self, inputs):
        outputs = self.conv(inputs)
        if self.do_batch:
            outputs = self.bn(outputs)  
        outputs=F.relu(outputs)        
        if self.dov>0:
            outputs = self.do(outputs)
        
        return outputs

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

class DilatedLSTMResC(torch.nn.Module):
    def __init__(self):
        super(DilatedLSTMResC, self).__init__()

        self.dl1 = DilatedPropB(12, 24, dilations=[1, 2, 3, 4, 5, 7], num_feat_maps=6)
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
            torch.nn.Flatten(start_dim=1),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(72*9, 40, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(40, len(valid), bias=True),
            torch.nn.Softmax(dim=-1)
        )

        ## weigths initialization wih xavier method
        for i, m in enumerate(self.modules()):
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                init.constant_(m.bias, 0)

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
        x = F.avg_pool1d(x, kernel_size=2)

        x = self.dl7(x)
        x = self.dl8(x)
        x = F.avg_pool1d(x, kernel_size=2)

        x = self.dl9(x)
        x = F.avg_pool1d(x, kernel_size=3)

        return self.mlp(x)

class DilatedLSTMResD(torch.nn.Module):
    def __init__(self):
        super(DilatedLSTMResD, self).__init__()

        self.dl1 = DilatedPropB(12, 24, dilations=[1, 2, 3, 4, 5, 7], num_feat_maps=6)
        self.dl2 = DilatedPropB(24, 24, dilations=[1, 2, 3, 4, 5], num_feat_maps=8)
        self.dl3 = DilatedPropB(24, 32, dilations=[1, 2, 3, 4, 5], num_feat_maps=12)
        self.dl4 = DilatedPropB(32, 32, dilations=[1, 2, 3, 4, 5], num_feat_maps=12)

        self.dl5 = DilatedPropB(32, 48, dilations=[1, 2, 3, 4], num_feat_maps=14)
        self.dl6 = DilatedPropB(48, 48, dilations=[1, 2, 3, 4], num_feat_maps=14)
        self.dl7 = DilatedPropB(48, 48, dilations=[1, 2, 3, 4], num_feat_maps=14)
        self.dl8 = DilatedPropB(48, 48, dilations=[1, 2, 3, 4], num_feat_maps=14)

        self.dl9 = DilatedPropB(48, 64, dilations=[1, 2, 3], num_feat_maps=16)
        self.dl10 = DilatedPropB(64, 72, dilations=[1, 2, 3], num_feat_maps=32)
        self.dl11 = DilatedPropB(72, 72, dilations=[1, 2, 3], num_feat_maps=32)

        self.dl12 = DilatedPropB(72, 72, dilations=[1, 2], num_feat_maps=42)
        self.dl13 = DilatedPropB(72, 72, dilations=[1], num_feat_maps=48)

        self.mlp = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=1),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(72*2, 64, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(64, len(valid), bias=True),
            torch.nn.Softmax(dim=-1)
        )

        ## weigths initialization wih xavier method
        for i, m in enumerate(self.modules()):
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                init.constant_(m.bias, 0)

    def forward(self, x):
        if len(x.shape) != 3:
            x = x.unsqueeze(dim=1)

        x = self.dl1(x)
        x = self.dl2(x)
        x = F.avg_pool1d(x, kernel_size=3)

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
        x = self.dl10(x)
        x = F.avg_pool1d(x, kernel_size=3)

        x = self.dl11(x)
        x = self.dl12(x)
        x = self.dl13(x)
        x = F.avg_pool1d(x, kernel_size=3)

        return self.mlp(x)

class DilatedLSTMResE(torch.nn.Module):
    def __init__(self):
        super(DilatedLSTMResE, self).__init__()

        self.dl1 = DilatedPropB(12, 36, dilations=[1, 2, 3, 4, 5, 7], num_feat_maps=12)
        self.dl2 = DilatedPropB(36, 48, dilations=[1, 2, 3, 4, 5], num_feat_maps=16)
        self.dl3 = DilatedPropB(48, 48, dilations=[1, 2, 3, 4, 5], num_feat_maps=16)
        self.dl4 = DilatedPropB(48, 64, dilations=[1, 2, 3, 4, 5], num_feat_maps=24)

        self.dl5 = DilatedPropB(64, 72, dilations=[1, 2, 3, 4], num_feat_maps=24)
        self.dl6 = DilatedPropB(72, 84, dilations=[1, 2, 3, 4], num_feat_maps=24)
        
        self.dl7 = DilatedPropB(84, 96, dilations=[1, 2, 3], num_feat_maps=48)
        self.dl8 = DilatedPropB(96, 108, dilations=[1, 2, 3], num_feat_maps=48)

        self.dl9 = DilatedPropB(108, 128, dilations=[1, 2], num_feat_maps=64)
        self.dl10 = DilatedPropB(128, 128, dilations=[1], num_feat_maps=128)

        self.mlp = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=1),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(128*2, 64, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(64, len(valid), bias=True),
            torch.nn.Softmax(dim=-1)
        )
        ## weigths initialization wih xavier method
        for i, m in enumerate(self.modules()):
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                init.constant_(m.bias, 0)

    def forward(self, x):
        if len(x.shape) != 3:
            x = x.unsqueeze(dim=1)

        x = self.dl1(x)
        x = self.dl2(x)
        x = F.avg_pool1d(x, kernel_size=3)

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
        x = F.avg_pool1d(x, kernel_size=3)

        return self.mlp(x)

class DilatedLSTMResE(torch.nn.Module):
    def __init__(self):
        super(DilatedLSTMResE, self).__init__()

        self.dl1 = DilatedPropB(12, 36, dilations=[1, 2, 3, 4, 5, 7], num_feat_maps=12)
        self.dl2 = DilatedPropB(36, 48, dilations=[1, 2, 3, 4, 5], num_feat_maps=16)
        self.dl3 = DilatedPropB(48, 48, dilations=[1, 2, 3, 4, 5], num_feat_maps=16)
        self.dl4 = DilatedPropB(48, 64, dilations=[1, 2, 3, 4, 5], num_feat_maps=24)

        self.dl5 = DilatedPropB(64, 72, dilations=[1, 2, 3, 4], num_feat_maps=24)
        self.dl6 = DilatedPropB(72, 84, dilations=[1, 2, 3, 4], num_feat_maps=24)
        
        self.dl7 = DilatedPropB(84, 96, dilations=[1, 2, 3], num_feat_maps=48)
        self.dl8 = DilatedPropB(96, 108, dilations=[1, 2, 3], num_feat_maps=48)

        self.dl9 = DilatedPropB(108, 128, dilations=[1, 2], num_feat_maps=64)
        self.dl10 = DilatedPropB(128, 128, dilations=[1], num_feat_maps=128)

        self.mlp = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=1),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(128*2, 64, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(64, len(valid), bias=True),
            torch.nn.Softmax(dim=-1)
        )
        ## weigths initialization wih xavier method
        for i, m in enumerate(self.modules()):
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                init.constant_(m.bias, 0)

    def forward(self, x):
        if len(x.shape) != 3:
            x = x.unsqueeze(dim=1)

        x = self.dl1(x)
        x = self.dl2(x)
        x = F.avg_pool1d(x, kernel_size=3)

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
        x = F.avg_pool1d(x, kernel_size=3)

        return self.mlp(x)

class DilatedLSTMResF(nn.Module):
    def __init__(self, class_dim=len(valid)):
        super(DilatedLSTMResF, self).__init__()
        self.encode = nn.Sequential(
            DilatedPropB(12, 24, dilations=[1, 2, 3, 5, 7, 9, 11], num_feat_maps=8),
            DilatedPropB(24, 36, dilations=[1, 2, 3, 5, 7, 9, 11], num_feat_maps=8),
            nn.AvgPool1d(kernel_size=2),
            DilatedPropB(36, 48, dilations=[1, 2, 3, 5, 7, 9], num_feat_maps=12),
            DilatedPropB(48, 64, dilations=[1, 2, 3, 5, 7, 9], num_feat_maps=12),
            nn.AvgPool1d(kernel_size=2),
            DilatedPropB(64, 72, dilations=[1, 2, 3, 5, 7], num_feat_maps=16),
            DilatedPropB(72, 84, dilations=[1, 2, 3, 5, 7], num_feat_maps=16),
            nn.AvgPool1d(kernel_size=3),
            DilatedPropB(84, 96, dilations=[1, 2, 3, 5], num_feat_maps=24),
            DilatedPropB(96, 108, dilations=[1, 2, 3, 5], num_feat_maps=24),
            nn.AvgPool1d(kernel_size=3),
            DilatedPropB(108, 128, dilations=[1, 2, 3], num_feat_maps=36),
            nn.AvgPool1d(kernel_size=3),
            DilatedPropB(128, 128, dilations=[1, 2, 3], num_feat_maps=36),
            nn.AvgPool1d(kernel_size=3),
            nn.Flatten(start_dim=1),
            nn.Linear(128*5, 128, bias=True),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, class_dim),
            nn.Softmax(dim=-1)
        )
        
        for i, m in enumerate(self.modules()):
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                init.constant_(m.bias, 0)

    def forward(self, x):
        return  self.encode(x)

def wce(res,lbls,w_positive_tensor,w_negative_tensor):
    ## weighted crossetropy - weigths are for positive and negative 
    res_c = torch.clamp(res,min=1e-6,max=1-1e-6)
            
    p1=lbls*torch.log(res_c)*w_positive_tensor
    p2=(1-lbls)*torch.log(1-res_c)*w_negative_tensor
    
    return -torch.mean(p1+p2)

# load the dataset
data_map = None
import pickle
with open('data_map.pkl', 'rb') as f:
    data_map = pickle.load(f)

modal = DilatedLSTMResF()
print('####### params: ', sum(p.numel() for p in modal.parameters() if p.requires_grad))
optimizer = torch.optim.Adam(modal.parameters(), lr=0.0001)
modal.cuda()

fold = 0
train_data = np.load('./folds/train_{}.npy'.format(fold))
test_data = np.load('./folds/test_{}.npy'.format(fold))

x_train, x_test = [], []

for recording in train_data:
    x_train += data_map[recording]
for recording in test_data:
    x_test += data_map[recording]

print('###### train on: {} test on: {}'.format(len(x_train), len(x_test)))

params = {'batch_size': 64,
          'shuffle': True,
          'num_workers': 8}

# Generators
training_set = Dataset(x_train)
training_generator = torch.utils.data.DataLoader(training_set, **params)

testing_set = Dataset(x_test)
testing_generator = torch.utils.data.DataLoader(testing_set, **params)
torch.autograd.set_detect_anomaly(True)

max_f1 = -np.inf

code = 'DilatedLSTMResF_250_{}_run1'.format(fold)

for epoch in range(100):
    print('\n############### epoch: {} of {}'.format(epoch, code))

    train_mats = []
    for i, (segment, y) in enumerate(training_generator):
        optimizer.zero_grad()

        y_pred = modal(segment.cuda())
        loss = F.cross_entropy(y_pred, y.long().cuda())
        loss.backward()

        optimizer.step()

        f1 = f1_score(y, y_pred.argmax(dim=-1).detach().cpu().numpy(), average='micro')
        train_mats.append([loss.item(), f1])

        if i == -100:
            break

        print('\r[{:04d}] train loss: {:.4f} f1: {:.4f}'.format(i, loss.item(), f1), end='')

    train_mats = np.array(train_mats)
    train_mats = np.mean(train_mats, axis=0)

    print('\n ######## train loss: {:.4} f1: {:.4f}'.format(train_mats[0], train_mats[1]))

    test_mats = []
    for i, (segment, y) in enumerate(testing_generator):
        y_pred = modal(segment.cuda())
        loss = F.cross_entropy(y_pred, y.long().cuda())

        f1 = f1_score(y, y_pred.argmax(dim=-1).detach().cpu().numpy(), average='micro')
        test_mats.append([loss.item(), f1])

        if i == -100:
            break

        print('\r[{:04d}] test loss: {:.4f} f1: {:.4f} max-f1: {:.4f}'.format(i, loss.item(), f1, max_f1), end='')

    test_mats = np.array(test_mats)
    test_mats = np.mean(test_mats, axis=0)

    print('\n ######## test loss: {:.4} f1: {:.4f}'.format(test_mats[0], test_mats[1]))

    if test_mats[-1] > max_f1:
        print('##### f1 improved from {:.4f} to {:.4f}'.format(max_f1, test_mats[-1]))
        max_f1 = test_mats[-1]
        torch.save(modal.state_dict(),'./modals/modal_{}.tm'.format(code))
