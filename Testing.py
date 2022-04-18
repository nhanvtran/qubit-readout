from numpy import*
from pylab import*
import matplotlib.pyplot as plt
from h5py import File
import sklearn
#STIXGeneral

font = {'family' : 'STIXGeneral',
        'weight' : 'normal',
        'size'   : 22}

matplotlib.rc('font', **font)
# from slab.dsfit import*
# from slab import*
import json
#%%
expt_name = 'histogram'
filelist = [7]

tags = ['']

rancut = [6,6]
for jj,i in enumerate(filelist):
    filename = str(i).zfill(5) + "_"+expt_name.lower()+".h5"

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
#%%
from sklearn import svm
from sklearn.model_selection import train_test_split

DataSet = np.zeros((3*sample,3))
for ii in range(3):
    DataSet[ii*sample:(ii+1)*sample,0] = IQsss[2*ii][::int(a_num/sample)]
    DataSet[ii*sample:(ii+1)*sample,1] = IQsss[2*ii+1][::int(a_num/sample)]
    DataSet[ii*sample:(ii+1)*sample,2] = ii
    pass
shuffle(DataSet)

Xdata = np.zeros((3*sample,2))
Ydata = np.zeros(3*sample)
Xdata[:,0] = DataSet[:,0]
Xdata[:,1] = DataSet[:,1]
Ydata[:] = DataSet[:,2]   
#%% Load model
import joblib
filename = 'knn_classifier_voting_v4.joblib.pkl'
SVC_model = joblib.load(filename)
#%%
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
predicted_labels = SVC_model.predict(Xdata)
cnf_matrix = confusion_matrix(Ydata, predicted_labels)
print(cnf_matrix)
class_names = ['g', 'e', 'f']
disp = plot_confusion_matrix(SVC_model, Xdata, Ydata, 
                                 display_labels=class_names,                                
                                 cmap=plt.cm.Blues,normalize= 'true')
plt.show()


cm = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
acc = cm.diagonal()

Pge = 0.03*(1-exp(-3/108))
Peg = 1-exp(-3/108)

g_acc = acc[0]+Pge
e_acc = acc[1]+Peg
f_acc = acc[2]

print(g_acc,e_acc,f_acc)

#%%
from sklearn.metrics import classification_report
print(classification_report(Ydata, predicted_labels))