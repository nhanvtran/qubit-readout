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
#%%
expt_name = 'histogram'
filelist = [8]

tags = ['']

rancut = [6,6]
for jj,i in enumerate(filelist):
    filename = "C:\\Users\\oyesi\\Desktop\\Dropbox\\Shared Projects\\Purdue_codes\\ML_for_quantum_experiments\\Qubit_readout\\" + str(i).zfill(5) + "_"+expt_name.lower()+".h5"

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
#%%
from matplotlib.pyplot import*
from matplotlib.cm import*

save_fig = False

### x,y are arbitrary I,Q values
x = linspace(-0.01,0.015, 55)
y = linspace(-0.02,0.01, 55)

colors = rainbow(linspace(0, 1, len(counts)))
colors = ['royalblue', 'lightcoral', 'paleturquoise']

readout_map = []
read_thresh = 1/2 # probaiblity of state has to be above threshold to register

figure()
for x_point in x:
    for y_point in y:
        probs_counts = []
        for ii in range(len(centers[0:3])):
            dist = sqrt((x_point-centers[ii][0])**2 + (y_point-centers[ii][1])**2) # distance between aritrary I,Q value and centers of (g,e,f)
            dist_index = argmin(abs(bins-dist)) # which histogram bin does distance belong to
            probs_counts.append(counts[ii][dist_index]) # score of (g,e,f) at the determined bin
        probs_counts = probs_counts/sum(probs_counts) # normalize 'score' from each (g,e,f) distribution to obtain probabilities
        ### determine which state (g,e,f) this particular I,Q value comes from
        value = 0 
        if probs_counts[0] > read_thresh:
            value = 0 # assign g
        elif probs_counts[1] > read_thresh:
            value = 1 # assign e
        elif probs_counts[2] > read_thresh:
            value = 2 # assign f
        scatter(x_point, y_point, color = colors[value], alpha =1.0, marker = 's')

for ii in [0,1,2]:
    scatter(centers[ii][0],centers[ii][1],color='k', marker = 'o',s = 100)
xlim(-0.01,0.015)
ylim(-0.02,0.01)
if save_fig:
    savefig('../figures/readout_map.pdf', format='pdf', dpi=1200)

colors = ['b', 'r', 'g']    
### plot I,Q data
figure()
for ii in [2,1,0]:
    plot(IQsss[2*ii][::int(a_num/sample)],IQsss[2*ii+1][::int(a_num/sample)],'.',color = colors[ii],alpha=1.0)
xlim(-0.01,0.015)
ylim(-0.02,0.01)
if save_fig:
    savefig('../figures/readout_hist.pdf', format='pdf', dpi=1200)
#%%
expt_name = 'histogram'
filelist = [7]

tags = ['']

rancut = [6,6]
for jj,i in enumerate(filelist):
    filename = "C:\\Users\\oyesi\\Desktop\\Dropbox\\Shared Projects\\Purdue_codes\\ML_for_quantum_experiments\\Qubit_readout\\" + str(i).zfill(5) + "_"+expt_name.lower()+".h5"

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
g_I = IQsss[0]
g_Q = IQsss[1]
e_I = IQsss[2]
e_Q = IQsss[3]
f_I = IQsss[4]
f_Q = IQsss[5]

g_qubit_state_prob_counts = []
e_qubit_state_prob_counts = []
f_qubit_state_prob_counts = []
for jj in range(len(g_I)):
    g_probs_counts = []
    e_probs_counts = []
    f_probs_counts = []
    for ii in range(len(centers[0:3])):
            dist = sqrt((g_I[jj]-centers[ii][0])**2 + (g_Q[jj]-centers[ii][1])**2)
            dist_index = argmin(abs(bins-dist))
            g_probs_counts.append(counts[ii][dist_index])
            
            dist = sqrt((e_I[jj]-centers[ii][0])**2 + (e_Q[jj]-centers[ii][1])**2)
            dist_index = argmin(abs(bins-dist))
            e_probs_counts.append(counts[ii][dist_index])
            
            dist = sqrt((f_I[jj]-centers[ii][0])**2 + (f_Q[jj]-centers[ii][1])**2)
            dist_index = argmin(abs(bins-dist))
            f_probs_counts.append(counts[ii][dist_index])
            
    g_probs_counts = g_probs_counts/sum(g_probs_counts)
    g_qubit_state_prob_counts.append(g_probs_counts)
    
    e_probs_counts = e_probs_counts/sum(e_probs_counts)
    e_qubit_state_prob_counts.append(e_probs_counts)
    
    f_probs_counts = f_probs_counts/sum(f_probs_counts)
    f_qubit_state_prob_counts.append(f_probs_counts)

g_state_prob_list = asarray(g_qubit_state_prob_counts).T[0]
e_state_prob_list = asarray(e_qubit_state_prob_counts).T[1]
f_state_prob_list = asarray(f_qubit_state_prob_counts).T[2]


g_counter = 0
e_counter = 0
f_counter = 0
readout_thresh = 1/2
for ii in range(len(g_state_prob_list)):
    if g_state_prob_list[ii] <= readout_thresh:
        g_counter = g_counter + 1
    if e_state_prob_list[ii] <= readout_thresh:
        e_counter = e_counter + 1
    if f_state_prob_list[ii] <= readout_thresh:
        f_counter = f_counter + 1

Pge = 0.03*(1-exp(-3/108))
Peg = 1-exp(-3/108)

# Pge=0
# Peg=0

g_inf = g_counter/len(g_state_prob_list) - Pge
e_inf = e_counter/len(e_state_prob_list) - Peg
f_inf = f_counter/len(f_state_prob_list)

g_inf_err = sqrt(g_counter)/len(g_state_prob_list)
e_inf_err = sqrt(e_counter)/len(e_state_prob_list)
f_inf_err = sqrt(f_counter)/len(f_state_prob_list)
        
print ("g fidelity: %.3f pm %.3f" %(1-g_inf, g_inf_err))
print ("e fidelity: %.3f pm %.3f" %(1-e_inf, e_inf_err))
print ("f fidelity: %.3f pm %.3f" %(1-f_inf, f_inf_err))

figure(figsize = (6,2))
plot(g_state_prob_list, '.')
axhline(readout_thresh)
title("Prob of qubit prepared in g, measured as g")

figure(figsize = (6,2))
plot(e_state_prob_list, '.')
axhline(readout_thresh)
title("Prob of qubit prepared in e, measured as e")

figure(figsize = (6,2))
plot(f_state_prob_list, '.')
axhline(readout_thresh)
title("Prob of qubit prepared in f, measured as f")