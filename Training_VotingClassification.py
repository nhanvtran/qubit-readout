from numpy import*
from pylab import*
import matplotlib.pyplot as plt
from h5py import File

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
    filename = "E:\\Dropbox\\Shared Projects\\Purdue_codes\\ML_for_quantum_experiments\\Qubit_readout\\" + str(i).zfill(5) + "_"+expt_name.lower()+".h5"

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
#%%
### save (g,e,f) median values and distance from center probability distributions
centers = [g_center,e_center,f_center]
counts = [counts_g,counts_e,counts_f]
bins = bins[:-1]

### plot digitized I,Q values (g=blue, e=red, f=green)
fig, ax = plt.subplots(figsize=(6, 6))
for ii in [0,1,2]:
    ax.plot(IQsss[2*ii][::int(a_num/sample)],IQsss[2*ii+1][::int(a_num/sample)],'.',color = colors[ii],alpha=0.2)
ax.set_xlim(x0g-ran/rancut[0],x0g+ran/rancut[0])
ax.set_ylim(y0g-ran/rancut[0],y0g+ran/rancut[0])


for ii in [0,1,2]:
    ax.errorbar(centers[ii][0],centers[ii][1],fmt = 'o',color='yellow',markersize=10)
plt.xlim([-0.01, 0.02])
plt.ylim([-0.02, 0.01])
ax.set_xlabel('I')
ax.set_ylabel('Q')
plt.show()
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
    
X_train, X_test, Y_train, Y_test = train_test_split(Xdata, Ydata, random_state=0,test_size=0.2)

from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix

# training a KNN classifier 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import VotingClassifier

# define the base models
models = list()
models.append(('knn25', KNeighborsClassifier(n_neighbors=25)))
# models.append(('SVC', svm.SVC(kernel='sigmoid')))
models.append(('tree', DecisionTreeClassifier(max_depth = 7)))
# define the hard voting ensemble
ensemble = VotingClassifier(estimators=models, voting='soft',weights=[10,1]).fit(X_train, Y_train) 

predicted_labels = ensemble.predict(X_test)
cnf_matrix = confusion_matrix(Y_test, predicted_labels)
print(cnf_matrix)
disp = plot_confusion_matrix(ensemble, X_test, Y_test,                                 
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
#%% save SVC model 
import joblib
import pickle

filename = 'E:\\Dropbox\\Shared Projects\\Purdue_codes\\ML_for_quantum_experiments\\Qubit_readout\\knn_classifier_voting_v4.joblib.pkl'
_ = joblib.dump(ensemble, filename, compress=0)
