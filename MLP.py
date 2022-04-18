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

if __name__ == "__main__":
    cuda = torch.device('cuda')
    
    plt.close('all')
    # clear_all()
     
    start = time.time()
    
    Visuals = False
#STIXGeneral

    font = {'family' : 'STIXGeneral',
            'weight' : 'normal',
            'size'   : 22}
    
    matplotlib.rc('font', **font)
    # from slab.dsfit import*
    # from slab import*
    import json
    
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
            print ('Readout length = ',readout_length)
            print ('Readout window = ',window)
            print ("Digital atten = ",atten)
            print ("Readout Freq = ",freq)
            I = array(a['I'])
            Q = array(a['Q'])
            sample = a_num
            
            I,Q = I/2**15*ran,Q/2**15*ran
            
            colors = ['r','b','g']
            labels= ['g','e','f']
            titles=['I','Q']
    
            IQs = median(I[::3],1),median(Q[::3],1),median(I[1::3],1),median(Q[1::3],1),median(I[2::3],1),median(Q[2::3],1)
            IQsss = I.T.flatten()[0::3],Q.T.flatten()[0::3],I.T.flatten()[1::3],Q.T.flatten()[1::3],I.T.flatten()[2::3],Q.T.flatten()[2::3]
            
            fig = plt.figure(figsize=(15,7*5))
    
            x0g,y0g  = mean(IQsss[0][::int(a_num/sample)]),mean(IQsss[1][::int(a_num/sample)])
            x0e,y0e  = mean(IQsss[2][::int(a_num/sample)]),mean(IQsss[3][::int(a_num/sample)])
            x0f,y0f  = mean(IQsss[4][::int(a_num/sample)]),mean(IQsss[5][::int(a_num/sample)])
    
    colors = ['b', 'r', 'g']
    
    ### Find mmedian I,Q for each measured state (g,e,f)
    g_center = (median(IQsss[0]),median(IQsss[1]))
    e_center = (median(IQsss[2]),median(IQsss[3]))
    f_center = (median(IQsss[4]),median(IQsss[5]))
    
    
    ### Distance between each I,Q point and median of (g,e,f)
    gg_radius = sqrt((IQsss[0]-g_center[0])**2 + (IQsss[1]-g_center[1])**2)
    ee_radius = sqrt((IQsss[2]-e_center[0])**2 + (IQsss[3]-e_center[1])**2)
    ff_radius = sqrt((IQsss[4]-f_center[0])**2 + (IQsss[5]-f_center[1])**2)
    
    ### set histogram binning
    num_bins = int(sqrt(len(gg_radius)))+1
    bins = linspace(0,max(gg_radius),num_bins)
    
    if Visuals:
    ### histogram measured (g,e,f) distances from median
        fig = plt.figure(figsize = (20,5))
        ax = fig.add_subplot(1,4,1)
        ax.set_xlabel('Distance from g_center')
        ax.set_title('prep g')
        counts, bin_edges = np.histogram(gg_radius, bins)
        counts_g = counts/sum(counts) # normalize histogram values to obtain frequencies/probabilities
        bins_g = bin_edges[:-1]
        plot(bins_g, counts_g, 'o-', color = colors[0])
        
        
        
        ax = fig.add_subplot(1,4,2)
        ax.set_xlabel('Distance from e_center')
        ax.set_title('prep e')
        counts, bin_edges = np.histogram(ee_radius, bins)
        counts_e = counts/sum(counts)
        bins_e = bin_edges[:-1]
        plot(bins_e, counts_e, 'o-', color = colors[1])
        
        
        ax = fig.add_subplot(1,4,3)
        ax.set_xlabel('Distance from f_center')
        ax.set_title('prep f')
        counts, bin_edges = np.histogram(ff_radius, bins)
        counts_f = counts/sum(counts)
        bins_f = bin_edges[:-1]
        plot(bins_f, counts_f, 'o-', color = colors[2])
        
        ### save (g,e,f) median values and distance from center probability distributions
        centers = [g_center,e_center,f_center]
        counts = [counts_g,counts_e,counts_f]
        bins = bins[:-1]
        
        ### plot digitized I,Q values (g=blue, e=red, f=green)
        ax = fig.add_subplot(1,4,4)
        for ii in [0,1,2]:
            ax.plot(IQsss[2*ii][::int(a_num/sample)],IQsss[2*ii+1][::int(a_num/sample)],'.',color = colors[ii],alpha=0.2)
        ax.set_xlim(x0g-ran/rancut[0],x0g+ran/rancut[0])
        ax.set_ylim(y0g-ran/rancut[0],y0g+ran/rancut[0])
        
        
        for ii in [0,1,2]:
            ax.errorbar(centers[ii][0],centers[ii][1],fmt = 'o',color='yellow',markersize=10)
        
        ax.set_xlabel('I')
        ax.set_ylabel('Q')
        # ax.cla()  
    
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
            self.data[:,0] = self.DataSet[:,0]
            self.data[:,1] = self.DataSet[:,1]
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
                # x = self.dropout(x)
                x = F.relu(self.fc2(x))
                x = F.relu(self.fc3(x))
                x = F.relu(self.fc4(x))
                x = F.relu(self.fc5(x))
                # x = self.dropout(x)
                # x = F.relu(self.fc3(x))
                # x = self.fc2(x)
                # x = self.fc4(x)
                return x
    
    model = Net()
    model = model.to(cuda)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    
    train_loss_track=np.array([])

    n_epochs = 100
    for epoch in tqdm(range(n_epochs), desc='Epoch'):
        cc = 0
        train_loss = 0
        model.train() # prep model for training
    
        for data, label in train_loader:
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            data = data.to(cuda)
            # data_short=torch.unsqueeze(data_short, 1)
            output = model(data)
            output = torch.squeeze(output, 1)
            label = label.to(cuda)
            loss = criterion(output, label.long())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
           
        train_loss_track=np.append(train_loss_track,np.asarray(train_loss))
        # print('Epoch ',epoch,' Training Loss: ', train_loss_track)
    
    
    # plt.pause(0.0001)
    plt.plot(train_loss_track)
    plt.xlabel('Epochs')
    plt.title("Training Loss")
    plt.figure(figsize = (12,7))
    plt.show()
    
    # torch.save(model.state_dict(), 'model.pt')
    # model.load_state_dict(torch.load('model.pt'))
    # model = model.to(cuda)
    # # initialize lists to monitor test loss and accuracy
    # test_loss = 0.0
    # model.eval() # prep model for evaluation
    # cc = 0
    # y_pred = []
    # y_true = []
    # for data, target in test_loader:
    #     data=data.to(cuda)
    #     output = model(data)
    #     output = torch.squeeze(output, 1)
        
    #     target = target.to(cuda) 
    #     loss = criterion(torch.squeeze(output), target.long())
    #     output = torch.unsqueeze(output, 1)
    #     output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
    #     y_pred.extend(output) # Save Prediction
        
    #     target = target.data.cpu().numpy()
    #     y_true.extend(target) # Save Truth

    #     test_loss += loss.item()*data.size(0)
    #     test_loss += loss.item()
        
    # classes = ('g', 'e', 'f')
    
    # # y_true = int64(y_true)
    # # cf_matrix = confusion_matrix(y_true, y_pred)
    # # df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *10, index = [i for i in classes],
    # #                       columns = [i for i in classes])
    # # plt.figure(figsize = (12,7))
    # # sn.heatmap(df_cm, annot=True)
    # # plt.savefig('output.png')
    
    # print('Test Loss: ', test_loss)
    
    end = time.time()
    print('Total Time Elapsed:', end - start, 'seconds')