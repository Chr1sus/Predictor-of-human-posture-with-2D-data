import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import random
import math
import time
import h5py

def get_data(path):
	os.chdir(path)
	info = []
	w=[]
	print('reading encoded data')
	for file in os.listdir():
		input=np.load(file)
		s,t,d=np.shape(input)
		if d==4:
			rest=np.ones((128,8))*-55
			for n in range(t):
				rest[n][:4]=input[0][n]
			w.append(rest)
		else:
			w.append(input[0])
		#w=torch.load(file)
		#w=w.detach().cpu().numpy()
		#w=np.transpose(w[0])
		info.append(w)
		w=[]
	return info


def norm_matrix(datain):
    ns,_,_,_=np.shape(datain)
    scaled_in=np.copy(datain)
    #scaled_out=np.copy(dataout)
    #data=np.append(datain,dataout,axis=0)
    data=datain
    s,n,t,d=np.shape(data)
    print('dimensions',s,' ',n,' ',t,' ',d)
    max_now=0
    min_now=0
    for m in range(s):
        max_past=data[m][0].max()
        min_past=data[m][0].min()
        if max_past>max_now:
            max_now=max_past
        if min_past<min_now:
            min_now=min_past
    print('maximum value',max_now)
    print('minimum value',min_now)
    opr=np.ones((t,d))
    for m in range(ns):
        scaled_in[m][0] = (datain[m][0]-opr*min_now)/(max_now-min_now)*10000
        #scaled_out[m][0] = (dataout[m][0]-opr*min_now)/(max_now-min_now)*10000

    #return scaled_in,scaled_out,max_now,min_now
    return scaled_in,max_now,min_now

def des_norm_matrix(datain,max,min):
    ns,_,_,_=np.shape(datain)
    scaled_in=np.copy(datain)
    #scaled_out=np.copy(dataout)
    #data=np.append(datain,dataout,axis=0)
    data=datain
    s,n,t,d=np.shape(data)
    print('dimensions',s,' ',n,' ',t,' ',d)
    opr=np.ones((t,d))
    for m in range(ns):
        scaled_in[m][0] = (datain[m][0]*(max-min)/10000)+opr*min
        #scaled_in[m][0] = (datain[m][0]-opr*min_now)/(max_now-min_now)*10000
        #scaled_out[m][0] = (dataout[m][0]-opr*min_now)/(max_now-min_now)*10000

    #return scaled_in,scaled_out,max_now,min_now
    return scaled_in

def transp(data):
    ns,m,hg,wd=np.shape(data)
    print('dimensions to transpose',ns,m,hg,wd)
    out=np.zeros((ns,wd,hg,m))
    for i in range(ns):
        out[i]=np.transpose(data[i])
    return out

def des_transp(data):
    ns,m,hg,wd=np.shape(data)
    print('dimensions to transpose',ns,m,hg,wd)
    out=np.zeros((ns,m,wd,hg))
    for i in range(ns):
        out[i]=np.transpose(data[i],(0,2,1))
    return out

def process_data_for_aberman(data):
        samples,frames,joint2=np.shape(data)
        d=np.zeros((samples,frames,15,2))
        for n in range(samples):
            for f in range(frames):
                for j in range(int(round(joint2/2))):
                    d[n][f][j][0]=data[n][f][j*2]
                    d[n][f][j][1]=data[n][f][j*2+1]
        print('shape of data for aberman',np.shape(d))
        return d

def x_y_one(data):

    n=0
    x=[]
    y=[]

    while n<15:
        x=np.append(x,data[n][0])
        y=np.append(y,data[n][1])
   
        n+=1

    return x,y


def x_y_c(data,s,f):
    j=0
    n=0
    x=[]
    y=[]
    while j<64:
        if n<15:
            x=np.append(x,data[s][f][n][0])
            y=np.append(y,data[s][f][n][1])
        j+=1
        n+=1

    return x,y

def norm_m(datain,dataout):
    print('normalizing test')
    ns,_,_,_=np.shape(datain)
    scaled_in=np.copy(datain)
    scaled_out=np.copy(dataout)
    data=np.append(datain,dataout,axis=0)
    s,n,t,d=np.shape(data)
    print('dimensions',s,' ',n,' ',t,' ',d)
    max_now=0
    min_now=0
    for m in range(s):
        for j in range(n):
            max_past=data[m][j].max()
            min_past=data[m][j].min()
            if max_past>max_now:
                max_now=max_past
            if min_past<min_now:
                min_now=min_past
    print('maximum value',max_now)
    print('minimum value',min_now)
    opr=np.ones((t,d))
    for m in range(ns):
        for j in range(n):
            scaled_in[m][j] = (datain[m][j]-opr*min_now)/(max_now-min_now)
            scaled_out[m][j] = (dataout[m][j]-opr*min_now)/(max_now-min_now)


    return scaled_in,scaled_out,max_now,min_now


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
                TEX[j][0]=TEX[j][0]+MSE
 
    TEX=TEX/ns
    return TEX

def norm_from_spine(dat0,dat1,dat2,dat3,dat4):
    refx0=dat0[2][0]
    refy0=dat0[2][1]
    refx1=dat1[2][0]
    refy1=dat1[2][1]
    refx2=dat2[2][0]
    refy2=dat2[2][1]
    refx3=dat3[2][0]
    refy3=dat3[2][1]
    refx4=dat4[2][0]
    refy4=dat4[2][1]
    
    joints,xyc=np.shape(dat0)
    print('joints',joints,'xy',xyc)
    
    for ns in range(joints):
        dat0[ns][0]=dat0[ns][0]-refx0
        dat0[ns][1]=dat0[ns][1]-refy0
        dat1[ns][0]=dat1[ns][0]-refx1
        dat1[ns][1]=dat1[ns][1]-refy1
        dat2[ns][0]=dat2[ns][0]-refx2
        dat2[ns][1]=dat2[ns][1]-refy2
        dat3[ns][0]=dat3[ns][0]-refx3
        dat3[ns][1]=dat3[ns][1]-refy3
        dat4[ns][0]=dat4[ns][0]-refx4
        dat4[ns][1]=dat4[ns][1]-refy4

    x0,y0=x_y_one(dat0)
    x1,y1=x_y_one(dat1)
    x2,y2=x_y_one(dat2)
    x3,y3=x_y_one(dat3)
    x4,y4=x_y_one(dat4)
    print(x0)
    return x0,y0,x1,y1,x2,y2,x3,y3,x4,y4


def norm_from_spine2(dat0,dat1):
    refx0=dat0[2][0]
    refy0=dat0[2][1]
    refx1=dat1[2][0]
    refy1=dat1[2][1]
    
    joints,xyc=np.shape(dat0)
    print('joints',joints,'xy',xyc)
    
    for ns in range(joints):
        dat0[ns][0]=dat0[ns][0]-refx0
        dat0[ns][1]=dat0[ns][1]-refy0
        dat1[ns][0]=dat1[ns][0]-refx1
        dat1[ns][1]=dat1[ns][1]-refy1

    x0,y0=x_y_one(dat0)
    x1,y1=x_y_one(dat1)

    print(x0)
    return x0,y0,x1,y1



