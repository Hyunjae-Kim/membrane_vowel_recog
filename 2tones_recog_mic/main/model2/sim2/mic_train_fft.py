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
parser.add_argument('--point_len', type=int)
parser.add_argument('--ch_num', type=int)
args = parser.parse_args()

model_num = 2
data_num = 2
point_len = args.point_len
ch_num = args.ch_num
time_len =  point_len/10
point_len = int(point_len/2)+1
fc_len = int(point_len/32)+1


class Model(nn.Module):
    def __init__(self, loss):
        super(Model, self).__init__()
        input_c = ch_num
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
            nn.Linear(1024, 20),
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
        
        _, output_tr = torch.topk(output_tr, 2)
        output_tr, _ = output_tr.sort()
        _, target_tr = torch.topk(target_tr, 2)
        target_tr, _ = target_tr.sort()

        dloss_tr += loss_tr.cpu().item()
        dacc_tr += torch.sum(torch.sum(output_tr==target_tr, 1)==2).item()
        
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
        
        _, output_te = torch.topk(output_te, 2)
        output_te, _ = output_te.sort()
        _, target_te = torch.topk(target_te, 2)
        target_te, _ = target_te.sort()

        dloss_te += loss_te.cpu().item()
        dacc_te += torch.sum(torch.sum(output_te==target_te, 1)==2).item()
        
    te_loss.append(dloss_te/bat_te[0])
    te_acc.append(dacc_te/(bat_te[0]*bat_te[1]))
    return te_loss, te_acc


if __name__=='__main__':
    print('[Training]%dch_fft_%.1f_model%d'%(ch_num, time_len, model_num))
    torch.manual_seed(37)
    torch.cuda.manual_seed_all(37)
    torch.backends.cudnn.deterministic = True
    
    trainX = np.load('../../../npy_data/sim_data%d/%dch_fft/%dch_fft_%.1fms_trainX.npy'\
                     %(data_num, ch_num, ch_num, time_len))
    trainY = np.load('../../../npy_data/sim_data%d/%dch_fft/%dch_fft_%.1fms_trainY.npy'\
                     %(data_num, ch_num, ch_num, time_len))
    testX = np.load('../../../npy_data/sim_data%d/%dch_fft/%dch_fft_%.1fms_testX.npy'\
                    %(data_num, ch_num, ch_num, time_len))
    testY = np.load('../../../npy_data/sim_data%d/%dch_fft/%dch_fft_%.1fms_testY.npy'\
                    %(data_num, ch_num, ch_num, time_len))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    batch_num = int(len(testX)/2)
    batch_tr = [int(len(trainX)/batch_num), batch_num]
    batch_te = [int(len(testX)/batch_num), batch_num]
    
    trainX = torch.Tensor(trainX).to(device)
    trainY = torch.Tensor(trainY).to(device)
    testX = torch.Tensor(testX).to(device)
    testY = torch.Tensor(testY).to(device)
    print('%dch_fft data shape  -  %.1f ms'%(ch_num, time_len))
    print('train set :', np.shape(trainX) , np.shape(trainY))
    print('test set :', np.shape(testX) ,np.shape(testY))
    
    
    learning_rate = 0.00005
    loss_func=nn.BCELoss()
    model = nn.DataParallel(Model(loss_func)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=batch_num, eta_min = 3e-6)

#     model.load_state_dict(torch.load('../../../ckpt/model%d/sim_data%d/%dch_fft/%.1f_ckpt_1000.pt'\
#                        %(model_num, data_num, ch_num, time_len)))
    a = time.time()
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []
    for epoch in range(501):
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
            torch.save(model.state_dict(),'../../../ckpt/model%d/sim_data%d/%dch_fft/%.1f_ckpt2_%d.pt'\
                       %(model_num, data_num, ch_num, time_len, epoch))
            
            np.savetxt('../../../result/model%d/sim_data%d/%dch_fft/%.1f_loss_tr2.txt'\
                       %(model_num, data_num, ch_num, time_len), train_loss)
            np.savetxt('../../../result/model%d/sim_data%d/%dch_fft/%.1f_loss_te2.txt'\
                       %(model_num, data_num, ch_num, time_len), test_loss)
            np.savetxt('../../../result/model%d/sim_data%d/%dch_fft/%.1f_acc_tr2.txt'\
                       %(model_num, data_num, ch_num, time_len), train_acc)
            np.savetxt('../../../result/model%d/sim_data%d/%dch_fft/%.1f_acc_te2.txt'\
                       %(model_num, data_num, ch_num, time_len), test_acc)
        
        if test_acc[-1]>0.99:
            print('@@@@@@@ save model : epoch %d'% epoch)
            torch.save(model.state_dict(),'../../../ckpt/model%d/sim_data%d/%dch_fft/%.1f_ckpt2_%d.pt'\
                       %(model_num, data_num, ch_num, time_len, epoch))

            np.savetxt('../../../result/model%d/sim_data%d/%dch_fft/%.1f_loss_tr2.txt'\
                       %(model_num, data_num, ch_num, time_len), train_loss)
            np.savetxt('../../../result/model%d/sim_data%d/%dch_fft/%.1f_loss_te2.txt'\
                       %(model_num, data_num, ch_num, time_len), test_loss)
            np.savetxt('../../../result/model%d/sim_data%d/%dch_fft/%.1f_acc_tr2.txt'\
                       %(model_num, data_num, ch_num, time_len), train_acc)
            np.savetxt('../../../result/model%d/sim_data%d/%dch_fft/%.1f_acc_te2.txt'\
                       %(model_num, data_num, ch_num, time_len), test_acc)

            break
            
    print("training complete! - calculation time :", time.time()-a, '  seconds\n\n')