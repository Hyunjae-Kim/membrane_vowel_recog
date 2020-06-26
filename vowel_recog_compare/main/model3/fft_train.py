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
args = parser.parse_args()

data_type = 'fft'
model_num = 3
point_len = args.point_len
time_len =  point_len/10  ## ms
if time_len==0.5:
    point_len = 13
else:
    point_len = int(45000*time_len/1000/2)+1

class Model(nn.Module):
    def __init__(self, loss):
        super(Model, self).__init__()
        nodes = 256
        self.fc1 = nn.Sequential(
            nn.Linear(point_len, nodes*4),
            nn.LeakyReLU(negative_slope=0.1),
            nn.BatchNorm1d(nodes*4))
        self.fc2 = nn.Sequential(
            nn.Linear(nodes*4, nodes*8),
            nn.LeakyReLU(negative_slope=0.1),
            nn.BatchNorm1d(nodes*8))
        self.fc3 = nn.Sequential(
            nn.Linear(nodes*8, nodes*4),
            nn.LeakyReLU(negative_slope=0.1),
            nn.BatchNorm1d(nodes*4))
        self.fc4 = nn.Sequential(
            nn.Linear(nodes*4, nodes*2),
            nn.LeakyReLU(negative_slope=0.1),
            nn.BatchNorm1d(nodes*2))
        self.fc5 = nn.Sequential(
            nn.Linear(nodes*2, 5),
            nn.Softmax(dim=1))
        self.loss = loss
        
    def forward(self, data, target):
        x = self.fc1(data)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        h = self.fc5(x)
        
        l = self.loss(h, target)
        return l, h, target

def train(model, trainX, trainY, batch, device, optimizer, train_loss, train_acc):
    model.train()
    dloss = 0
    dacc = 0
    
    rand = torch.randperm(trainX.size()[0])
    trainX = trainX[rand]
    trainY = trainY[rand]
    
    for i in range(batch[0]):
        optimizer.zero_grad()
        loss, output, target = model(trainX[i*batch[1]:(i+1)*batch[1]], trainY[i*batch[1]:(i+1)*batch[1]])
        loss = loss.sum()
        loss.backward()
        optimizer.step()
        
        _, output = torch.max(output, 1)
        _, target = torch.max(target, 1)

        dloss += loss.cpu().item()
        dacc += (output==target).sum().item()
        
    train_loss.append(dloss/batch[0])
    train_acc.append(dacc/(batch[0]*batch[1]))
    return train_loss, train_acc

def test(model, testX, testY, device, test_loss, test_acc):
    model.eval()
    loss, output, target = model(testX, testY)
    loss = loss.sum()
    
    _, output = torch.max(output, 1)
    _, target = torch.max(target, 1)
    
    test_loss.append(loss.cpu().item())
    test_acc.append((output==target).sum().item()/len(testX))
    return test_loss, test_acc


if __name__=='__main__':    
    print('[Training]%s_%.1f_model%d'%(data_type, time_len, model_num))
    torch.manual_seed(37)
    torch.cuda.manual_seed_all(37)
    torch.backends.cudnn.deterministic = True
    
    trainX = np.load('../../npy_data/%s/%s_%.1fms_trainX.npy'%(data_type, data_type, time_len))
    trainY = np.load('../../npy_data/%s/%s_%.1fms_trainY.npy'%(data_type, data_type, time_len))
    testX = np.load('../../npy_data/%s/%s_%.1fms_testX.npy'%(data_type, data_type, time_len))
    testY = np.load('../../npy_data/%s/%s_%.1fms_testY.npy'%(data_type, data_type, time_len))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    batch_num = 1000
    batch = [int(len(trainX)/batch_num), batch_num]
    
    trainX = torch.Tensor(trainX).to(device)
    trainY = torch.Tensor(trainY).to(device)
    testX = torch.Tensor(testX).to(device)
    testY = torch.Tensor(testY).to(device)
    print('%s data shape  -  %.1f ms'%(data_type, time_len))
    print('train set :', np.shape(trainX) , np.shape(trainY))
    print('test set :', np.shape(testX) ,np.shape(testY))
    
    
    learning_rate = 0.00005
    loss_func=nn.BCELoss()
    model = nn.DataParallel(Model(loss_func)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=batch_num, eta_min = 3e-6)

#     model.load_state_dict(torch.load('ckpt/model%d_raw/%.1f_ckpt_2000.pt'%(model_num, time_len)))
    a = time.time()
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []
    for epoch in range(1001):
        train_loss, train_acc = train(model, trainX, trainY, batch, device, optimizer, train_loss, train_acc)
        test_loss, test_acc = test(model, testX, testY, device, test_loss, test_acc)
        scheduler.step()
        
        if epoch%10==0: 
            print('epoch %d - train loss : %.7f  /  test loss : %.7f'%(epoch, train_loss[-1], test_loss[-1]))
            print('           train acc : %.7f  /  test acc : %.7f'%(train_acc[-1], test_acc[-1]))
        if epoch%50==0:
            print('@@@@@@@ save model : epoch %d'% epoch)
            torch.save(model.state_dict(),'../../ckpt/model%d/%s/%.1f_ckpt_%d.pt'%(model_num, data_type, time_len, epoch))
            np.savetxt('../../result/model%d/%s/%.1f_loss_tr.txt'%(model_num, data_type, time_len), train_loss)
            np.savetxt('../../result/model%d/%s/%.1f_loss_te.txt'%(model_num, data_type, time_len), test_loss)
            np.savetxt('../../result/model%d/%s/%.1f_acc_tr.txt'%(model_num, data_type, time_len), train_acc)
            np.savetxt('../../result/model%d/%s/%.1f_acc_te.txt'%(model_num, data_type, time_len), test_acc)
    print("training complete! - calculation time :", time.time()-a, '  seconds\n\n')


