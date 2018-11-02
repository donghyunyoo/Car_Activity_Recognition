import sys
from myconfig import *
import pickle
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from model import Bid_RNN
import torch.backends.cudnn as cudnn
# cudnn.enabled=False

step = 150
def expand_feat(feat,lab):
    split_feat, split_lab = [],[]
    for t in range(0, len(lab), step):
        e = min(t+2*step, len(lab))
        split_feat.append(feat[t:e])
        split_lab.append(lab[t:e])
    return split_feat, split_lab


# model = Bid_RNN(2, 4, class_num=4).cuda()
# 3 correspond to the number of features or number of columns in final_feautures. and 4 correspond to the number of classes.
model = Bid_RNN(4, 4, class_num=4).cuda()
model.train()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

featlst,lablst = [], []
scene_2_train = ['VIRAT_S_040103_05_000729_000804', 'VIRAT_S_040103_07_001011_001093', 'VIRAT_S_040103_08_001475_001512',                          'VIRAT_S_040104_00_000120_000224','VIRAT_S_040104_01_000227_000457', 'VIRAT_S_040104_02_000459_000721',                          'VIRAT_S_040104_04_000854_000934','VIRAT_S_040104_05_000939_001116','VIRAT_S_040104_06_001121_001241',
                 'VIRAT_S_040104_09_001475_001583']

scene_1_train = ['VIRAT_S_000000','VIRAT_S_000001','VIRAT_S_000002','VIRAT_S_000005']

# for vid in scene_1_train + scene_2_train:
for vid in train_vid_list:
#     feats, labs = pickle.load(open('data/'+'VIRAT_S_040103_01_000132_000195.pkl','rb'), encoding='latin1')
    feats, labs = pickle.load(open('data/'+vid+'.pkl','rb'))
    for feat, lab in zip(feats,labs):
        if len(lab)>2*step:
            feat,lab = expand_feat(feat, lab)
            featlst+=feat
            lablst+=lab
        else:
            featlst.append(feat)
            lablst.append(lab)
print('tracklet num:', len(featlst))

# was 100 epochs earlier
for epoch in range(100):
    total_loss = 0
    for i in np.random.permutation(len(featlst)).tolist():
        feat, labs = featlst[i], lablst[i]
        feat = Variable(torch.from_numpy(feat).float().cuda())
        labs = Variable(torch.from_numpy(labs).long().cuda())
        feat = feat.unsqueeze(0)
        out = model(feat)
        loss = criterion(out, labs)
        total_loss+=loss.data[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
#     if epoch%2 == 0 and epoch !=0:
        
    if epoch%5 == 0 and epoch !=0:
        torch.save(model.state_dict(), 'rnn_state_all_epoch_'+ str(epoch) +'.pt')
    print('epoch: {}\t loss:{}'.format(epoch, total_loss/len(featlst)))

torch.save(model.state_dict(), 'rnn_state_all.pt')
