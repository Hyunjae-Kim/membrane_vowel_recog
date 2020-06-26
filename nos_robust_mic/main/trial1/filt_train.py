import os
import time
import shutil
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils as utils
import torch.nn.init as init
from torchvision import datasets

parser = argparse.ArgumentParser(description='..')
parser.add_argument('--SNR', type=int)
args = parser.parse_args()

data_type = 'filt'
SNR = args.SNR
point_len = 200
time_len =  20
fc_len = int(point_len/32)+1

class Model(nn.Module):
    def __init__(self, loss):
        super(Model, self).__init__()
        input_c = 1
        channel = 64
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_c, channel, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(negative_slope=0.1),
            nn.BatchNorm1d(channel))
        self.conv2 = nn.Sequential(
            nn.Conv1d(channel, channel*2, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(negative_slope=0.1),
            nn.BatchNorm1d(channel*2))
        self.conv3 = nn.Sequential(
            nn.Conv1d(channel*2, channel*2, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(negative_slope=0.1),
            nn.BatchNorm1d(channel*2))
        self.conv4 = nn.Sequential(
            nn.Conv1d(channel*2, channel*4, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(negative_slope=0.1),
            nn.BatchNorm1d(channel*4))
        self.conv5 = nn.Sequential(
            nn.Conv1d(channel*4, channel*4, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(negative_slope=0.1),
            nn.BatchNorm1d(channel*4))
        self.fc1 = nn.Sequential(
            nn.Linear(channel*4*fc_len, 1024),
            nn.LeakyReLU(negative_slope=0.1),
            nn.BatchNorm1d(1024))
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 5),
            nn.Softmax(dim=1))
        self.loss = loss
        
    def forward(self, data, target):
        x = self.conv1(data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x).contiguous()
        x = x.view(x.size()[0],-1)
        x = self.fc1(x)
        h = self.fc2(x)
        
        l = self.loss(h, target)
        return l, h, target


def train(mod_tr, trX, trY, bat_tr, dev, opt_tr, tr_loss, tr_acc):
    mod_tr.train()
    dloss_tr = 0
    dacc_tr = 0
    
    rand = torch.randperm(trX.size()[0])
    trX = trX[rand]
    trY = trY[rand]
    
    for i in range(bat_tr[0]):
        opt_tr.zero_grad()
        loss_tr, output_tr, target_tr = mod_tr(trX[i*bat_tr[1]:(i+1)*bat_tr[1]], 
                                            trY[i*bat_tr[1]:(i+1)*bat_tr[1]])
        loss_tr = loss_tr.sum()
        loss_tr.backward()
        opt_tr.step()
        
        _, output_tr = torch.max(output_tr, 1)
        _, target_tr = torch.max(target_tr, 1)

        dloss_tr += loss_tr.cpu().item()
        dacc_tr += (output_tr==target_tr).sum().item()
        
    tr_loss.append(dloss_tr/bat_tr[0])
    tr_acc.append(dacc_tr/(bat_tr[0]*bat_tr[1]))
    return tr_loss, tr_acc

def test(mod_te, teX, teY, bat_te, dev, te_loss, te_acc):
    mod_te.eval()
    dloss_te = 0
    dacc_te = 0
    
    for i in range(bat_te[0]):
        loss_te, output_te, target_te = mod_te(teX[i*bat_te[1]:(i+1)*bat_te[1]], 
                                              teY[i*bat_te[1]:(i+1)*bat_te[1]])
        loss_te = loss_te.sum()
        
        _, output_te = torch.max(output_te, 1)
        _, target_te = torch.max(target_te, 1)

        dloss_te += loss_te.cpu().item()
        dacc_te += (output_te==target_te).sum().item()
        
    te_loss.append(dloss_te/bat_te[0])
    te_acc.append(dacc_te/(bat_te[0]*bat_te[1]))
    return te_loss, te_acc


if __name__=='__main__':
    print('[Training]%s_%ddb_%.1fms'%(data_type, SNR, time_len))
    torch.manual_seed(37)
    torch.cuda.manual_seed_all(37)
    torch.backends.cudnn.deterministic = True
    
    trainX = np.load('../../npy_data/%s/%ddb/%ddb_%s_%.1fms_trainX.npy'\
                     %(data_type, SNR, SNR, data_type, time_len)).reshape(-1,1,point_len)
    trainY = np.load('../../npy_data/%s/%ddb/%ddb_%s_%.1fms_trainY.npy'\
                     %(data_type, SNR, SNR, data_type, time_len))
    testX = np.load('../../npy_data/%s/%ddb/%ddb_%s_%.1fms_testX.npy'\
                    %(data_type, SNR, SNR, data_type, time_len)).reshape(-1,1,point_len)
    testY = np.load('../../npy_data/%s/%ddb/%ddb_%s_%.1fms_testY.npy'\
                    %(data_type, SNR, SNR, data_type, time_len))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    batch_num = int(len(testX)/3)
    batch_tr = [int(len(trainX)/batch_num), batch_num]
    batch_te = [int(len(testX)/batch_num), batch_num]
    
    trainX = torch.Tensor(trainX).to(device)
    trainY = torch.Tensor(trainY).to(device)
    testX = torch.Tensor(testX).to(device)
    testY = torch.Tensor(testY).to(device)
    print('%ddb %s data shape  -  %.1f ms'%(SNR, data_type, time_len))
    print('train set :', np.shape(trainX) , np.shape(trainY))
    print('test set :', np.shape(testX) ,np.shape(testY))
    
    
    learning_rate = 0.0005
    loss_func=nn.BCELoss()
    model = nn.DataParallel(Model(loss_func)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=batch_num, eta_min = 3e-6)

#     model.load_state_dict(torch.load('ckpt/model%d_mem/%.1f_ckpt_2000.pt'%(model_num, time_len)))
    a = time.time()
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []
    for epoch in range(101):
        train_loss, train_acc = train(model, trainX, trainY, batch_tr, device, 
                                      optimizer, train_loss, train_acc)
        test_loss, test_acc = test(model, testX, testY, batch_te, device, 
                                   test_loss, test_acc)
        scheduler.step()
        
        if epoch%10==0: 
            print('epoch %d - train loss : %.7f  /  test loss : %.7f'\
                  %(epoch, train_loss[-1], test_loss[-1]))
            print('           train acc : %.7f  /  test acc : %.7f'\
                  %(train_acc[-1], test_acc[-1]))
            
        if epoch%100==0:
            print('@@@@@@@ save model : epoch %d'% epoch)
            torch.save(model.state_dict(),'../../ckpt/trial1/%s/%ddb_%s_%.1f_ckpt_%d.pt'\
                       %(data_type, SNR, data_type, time_len, epoch))
            
            np.savetxt('../../result/trial1/%s/%ddb_%s_%.1f_loss_tr.txt'\
                       %(data_type, SNR, data_type, time_len), train_loss)
            np.savetxt('../../result/trial1/%s/%ddb_%s_%.1f_loss_te.txt'\
                       %(data_type, SNR, data_type, time_len), test_loss)
            np.savetxt('../../result/trial1/%s/%ddb_%s_%.1f_acc_tr.txt'\
                       %(data_type, SNR, data_type, time_len), train_acc)
            np.savetxt('../../result/trial1/%s/%ddb_%s_%.1f_acc_te.txt'\
                       %(data_type, SNR, data_type, time_len), test_acc)
            
            
    print("training complete! - calculation time :", time.time()-a, '  seconds\n\n')