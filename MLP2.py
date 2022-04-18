from numpy import*
from pylab import*
import matplotlib.pyplot as plt
from h5py import File
import torch
import time
import math
import torch
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision import datasets
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import SubsetRandomSampler
from torch.nn.functional import normalize
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import torch.nn as nn
import torch.nn.functional as F
from sklearn import svm
from sklearn.model_selection import train_test_split
import seaborn as sn
import json
from sklearn.metrics import confusion_matrix

if __name__ == "__main__":
    cuda = torch.device('cuda')
    plt.close('all')
    start = time.time()
    Visuals = False
    
    expt_name = 'histogram'
    filelist = [8]
    tags = ['']
    rancut = [6,6]
    
    for jj,i in enumerate(filelist):
        filename =  str(i).zfill(5) + "_"+expt_name.lower()+".h5"
        with File(filename,'r') as a:
    
            hardware_cfg =  (json.loads(a.attrs['hardware_cfg']))
            experiment_cfg =  (json.loads(a.attrs['experiment_cfg']))
            quantum_device_cfg =  (json.loads(a.attrs['quantum_device_cfg']))
            ran = hardware_cfg['awg_info']['keysight_pxi']['m3102_vpp_range']
    
            expt_cfg = (json.loads(a.attrs['experiment_cfg']))[expt_name.lower()]
            numbins = expt_cfg['numbins']
            print (numbins)
            numbins = 200
            a_num = expt_cfg['acquisition_num']
            ns = expt_cfg['num_seq_sets']
            readout_length = quantum_device_cfg['readout']['length']
            window = quantum_device_cfg['readout']['window']
            atten = quantum_device_cfg['readout']['dig_atten']
            freq = quantum_device_cfg['readout']['freq']
            # print ('Readout length = ',readout_length)
            # print ('Readout window = ',window)
            # print ("Digital atten = ",atten)
            # print ("Readout Freq = ",freq)
            I = array(a['I'])
            Q = array(a['Q'])
            sample = a_num
            
            I,Q = I/2**15*ran,Q/2**15*ran
            
            colors = ['r','b','g']
            labels= ['g','e','f']
            titles=['I','Q']
    
            IQs = median(I[::3],1),median(Q[::3],1),median(I[1::3],1),median(Q[1::3],1),median(I[2::3],1),median(Q[2::3],1)
            IQsss = I.T.flatten()[0::3],Q.T.flatten()[0::3],I.T.flatten()[1::3],Q.T.flatten()[1::3],I.T.flatten()[2::3],Q.T.flatten()[2::3]
    
    IQsss_t=torch.tensor(IQsss)
    from torch.utils.data import Dataset, DataLoader
    class Qubit_Readout_Dataset(Dataset):
        def __init__(self):
            self.labels = torch.zeros(3*sample)
            self.data = torch.zeros(3*sample,2)
            self.DataSet = torch.zeros((3*sample,3))
            for ii in range(3):
                self.DataSet[ii*sample:(ii+1)*sample,0] = IQsss_t[2*ii][::int(a_num/sample)]
                self.DataSet[ii*sample:(ii+1)*sample,1] = IQsss_t[2*ii+1][::int(a_num/sample)]
                self.DataSet[ii*sample:(ii+1)*sample,2] = ii
            self.data[:,0] = self.DataSet[:,0]/torch.max(self.DataSet[:,0])
            self.data[:,1] = self.DataSet[:,1]/torch.max(self.DataSet[:,1])
            self.labels[:] = self.DataSet[:,2]
            # self.labels = normalize(self.labels, p=2)
            # self.data = normalize(self.data, p=2)
        
    
        def __len__(self):
            return self.labels.shape[0]
    
        def __getitem__(self, idx):
            return self.data[idx], self.labels[idx]

    dataset = Qubit_Readout_Dataset()
    
    train_data, test_data = torch.utils.data.random_split(dataset, [8000, 1000])
    
    transforms = torch.nn.Sequential(
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
    
    num_workers = 0
    batch_size = 64
        
    train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, 
                                                num_workers = num_workers, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size = batch_size, 
                                                num_workers = num_workers, shuffle=True)
    
    class Net(nn.Module):
            def __init__(self):
                super(Net,self).__init__()
                self.fc1 = nn.Linear(2, 1024)
                self.fc2 = nn.Linear(1024,1024)
                self.fc3 = nn.Linear(1024,512)
                self.fc4 = nn.Linear(512,64)
                self.fc5 = nn.Linear(64,3)
                # self.dropout = nn.Dropout(p=0.2)
                
            def forward(self,x):
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = F.relu(self.fc3(x))
                x = F.relu(self.fc4(x))
                x = F.relu(self.fc5(x))
                # x = self.dropout(x)
                return x
    
    model = Net()
    model = model.to(cuda)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    train_loss_track=np.array([])

    n_epochs = 20
    for epoch in tqdm(range(n_epochs), desc='Epoch'):
        cc = 0
        train_loss = 0
        model.train() # prep model for training
    
        for data, label in train_loader:
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            data = data.to(cuda)
            output = model(data)
            label = label.to(cuda)
            loss = criterion(output, label.long())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
           
        train_loss_track=np.append(train_loss_track,np.asarray(train_loss))

    plt.plot(train_loss_track)
    plt.xlabel('Epochs')
    plt.title("Training Loss")
    plt.figure(figsize = (12,7))
    plt.show()
    
    # torch.save(model.state_dict(), 'model.pt')
    # model.load_state_dict(torch.load('model.pt'))
    # model = model.to(cuda)
    
    # initialize lists to monitor test loss and accuracy
    test_loss = 0.0
    model.eval() 
    cc = 0
    y_true = torch.tensor([])
    y_true = y_true.to(cuda)
    y_pred = torch.tensor([])
    y_pred = y_pred.to(cuda)
    with torch.no_grad():
        for data, target in test_loader:
            
            data=data.to(cuda)
            output = model(data)
            target = target.to(cuda) 
            loss = criterion(output, target.long())
    
            test_loss += loss.item()*data.size(0)
            test_loss += loss.item()
            
            val, ind = torch.max(output,1)
            y_pred = torch.cat((y_pred, ind), 0)
            y_true = torch.cat((y_true, target), 0)

    acc = y_true-y_pred
    accuracy = (len(y_true)-torch.count_nonzero(acc))/len(y_true)
    accuracy = accuracy.item()
    
    print('Confusion Matrix:')
    print(confusion_matrix(y_pred.cpu(),y_true.cpu()))
    print('Test Accuracy: %', accuracy*100)
    # print('Test Loss: ', test_loss)
    
    end = time.time()
    print('Total Time Elapsed:', end - start, 'seconds')