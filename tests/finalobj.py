from pickle import NONE
from tkinter import Variable
from matplotlib.pyplot import axis
import torch
import numpy as np
import torch.nn
import os, sys
import time
import h5py

from ops.loaddata import load_data_qdrcn as ld
from ops.loaddata import load_data_ as ld1
from ops.loaddata import velocity as v
from ops.loaddata import norm
from scipy.ndimage.filters import gaussian_filter1d
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import csv
#import common as config
from ops.vizu import viz_gcn as viz
from ops.vizu import creat_skeleton, viz_aberman
import matplotlib.pyplot as plt
import ops.loaddata as ld
import matplotlib.image as mpimg

from Models.GCN.GCN_DCT import get_dct_matrix,AccumLoss,mpjpe_error_p3d,GCN,lr_decay

target_aberman=np.load('target_data_dir')
trupred=np.load('retrained_model')
print('trupred shape,',np.shape(trupred))
samp,fram,_,_=np.shape(target_aberman)
for n in range(samp):
    for fr in range(fram):
        target_aberman[n][fr]=np.interp(target_aberman[n][fr], (target_aberman[n][fr].min(), target_aberman[n][fr].max()), (-0.1, +0.1))

samp,fram,_,_=np.shape(trupred)
for n in range(samp):
    for fr in range(fram):
        trupred[n][fr]=np.interp(trupred[n][fr], (trupred[n][fr].min(), trupred[n][fr].max()), (-0.1, +0.1))

def calculate_error_per_frame(ns,fr,inp,obj,train_set):
    
    TEX=np.zeros((fr,1))
    for m in range(ns):
        for j in range(fr):
            if train_set is not None:
                fs=train_set[str(m)]
                MSE1 = np.square(np.subtract(inp[fs][j],obj[m][j])).mean()
                MSE2 = np.square(np.subtract(-inp[fs][j],obj[m][j])).mean()
                if MSE1>MSE2: MSE=MSE2
                else: MSE=MSE1
                TEX[j][0]=TEX[j][0]+MSE

            else:
                MSE = np.square(np.subtract(inp[m][j],obj[m][j])).mean()
                MSE2 = np.square(np.subtract(-inp[m][j],obj[m][j])).mean()
                if MSE>MSE2: MSE=MSE2
                TEX[j][0]=TEX[j][0]+MSE
 
    TEX=TEX/ns
    return TEX

inc_data=ld.dataset_for_poses(target_aberman)
print(np.shape(inc_data))
viz(inc_data[0][31])

inc_data=ld.dataset_for_poses(trupred)
viz(inc_data[25][0])
MSET=np.zeros(33)
MSE32=np.zeros(33)
for j in range(32):
    MS=calculate_error_per_frame(1,64,[trupred[j]],target_aberman,None)
    MSET[32-j]=MS.mean()
    MSE1 = np.square(np.subtract(-trupred[j][32],target_aberman[0][32])).mean()
    MSE2 = np.square(np.subtract(trupred[j][32],target_aberman[0][32])).mean()
    if MSE1>MSE2: MSE32[32-j]=MSE2
    else: MSE32[32-j]=MSE1

MSET[0]=0

frames=list(range(0,33))
plt.figure(1)
plt.plot(frames,MSET)
a, b = np.polyfit(frames, MSET, 1)
print(a)
plt.plot(frames, a*np.array(frames)+b, color='steelblue', linestyle='--', linewidth=2)
plt.title("MSE en el Aberman segun los cuadros predichos")
plt.xlabel("Cuadros predichos")
plt.ylabel("MSE promedio")
plt.xlim(0,32)
plt.show()


frames=list(range(0,33))
plt.figure(2)
plt.plot(frames,MSE32)
a, b = np.polyfit(frames, MSE32, 1)
print(a)
plt.plot(frames, a*np.array(frames)+b, color='steelblue', linestyle='--', linewidth=2)
plt.text(1, 0.0045, 'y = ' + '{:.5f}'.format(b) + ' + {:.5f}'.format(a) + 'x', size=14)
plt.title("MSE en el Aberman segun los cuadros predichos comparado con el cuadro 32")
plt.xlabel("Cuadros predichos")
plt.ylabel("MSE")
plt.xlim(0,32)
#plt.ylim(0,0.0051)
plt.show()