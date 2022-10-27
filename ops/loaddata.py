from binascii import Incomplete
import numpy as np
import os, sys
import warnings
from scipy.spatial import distance
import random
def testing_data_1(path):
    complete_test=[]
    incomplete_test=[]

    for (dirpath, dirnames, filenames)  in os.walk(path):
            for data in range(0,len(filenames),4):
                print(filenames[data])
                random_file=np.load(path + filenames[data])
                test=[]
                fr,_,_=np.shape(random_file)
                bg=round(fr/3)
                test=random_file[bg:(bg+64)]
                complete_test.append(test)
                incomplete_test.append(test[:32])

    return complete_test,incomplete_test
                
def dataset_for_poses(test):
    sample,sec,_,_=np.shape(test)
    out=np.zeros((sample,64,30))
    for ns in range(sample):
        for part in range(sec):
            row=[]
            j=0 #joint

            while j<15: 
                row=np.append(row,test[ns][part][j][0])
                row=np.append(row,test[ns][part][j][1])
                j+=1    
            out[ns][part]=row
    print('shape for testing',np.shape(out))

    return out

def load_data_(typ):
    path='/home/chrisus/Proyectofinal/Motion/processed/'
    subs = np.array([[1,3,4,5,6,7,9], [2], [11]])
    complete_data=[]
    E=[]
    G=[]
    T=[]
    DS=[]
    DR=[]
    PH=[]
    PO=[]
    PU=[]
    SI=[]
    SD=[]
    SM=[]
    WI=[]
    WD=[]
    WT=[]
    WL=[]
    aux=[]
    cont=0
    if typ=='train':
        ext=subs[0]
    if typ=='test':
        ext=subs[1]
    if typ=='val':
        ext=subs[2]
    for i in range(len(ext)):
        file=path + 'S' + str(ext[i])+'/'
        for (dirpath, dirnames, filenames)  in os.walk(file):
            for data in range(len(filenames)):
                N=450
                if filenames[data][0]=='E':
                    aux=obtain_data((file + filenames[data]),N)
                    E.append(aux)
                if filenames[data][0]=='G':
                    aux=obtain_data((file + filenames[data]),N)
                    G.append(aux)
                if filenames[data][0]=='T':
                    aux=obtain_data((file + filenames[data]),N)
                    T.append(aux)
                if filenames[data][0]=='D':
                    if filenames[data][2]=='s':
                        aux=obtain_data((file + filenames[data]),N)
                        DS.append(aux)
                    if filenames[data][2]=='r':
                        aux=obtain_data((file + filenames[data]),N)
                        DR.append(aux)
                if filenames[data][0]=='P':
                    if filenames[data][1]=='h':
                        aux=obtain_data((file + filenames[data]),N)
                        PH.append(aux)
                    if filenames[data][1]=='o':
                        aux=obtain_data((file + filenames[data]),N)
                        PO.append(aux)
                    if filenames[data][1]=='u':
                        aux=obtain_data((file + filenames[data]),N)
                        PU.append(aux)
                if filenames[data][0]=='S':
                    if filenames[data][1]=='m':
                        aux=obtain_data((file + filenames[data]),N)
                        SM.append(aux)
                    elif filenames[data][7]=='D' and filenames[data][1]=='i':
                        aux=obtain_data((file + filenames[data]),N)
                        SD.append(aux)
                    elif filenames[data][1]=='i':
                        aux=obtain_data((file + filenames[data]),N)
                        SI.append(aux)
                if filenames[data][0]=='W':
                    if filenames[data][4]=='i' and filenames[data][3]=='k' and filenames[data][7]!='D' :
                        aux=obtain_data((file + filenames[data]),N)
                        WI.append(aux)
                    elif filenames[data][7]=='D':
                        aux=obtain_data((file + filenames[data]),N)
                        WD.append(aux)
                    elif filenames[data][4]=='T': 
                        aux=obtain_data((file + filenames[data]),N)
                        WT.append(aux)
                    elif filenames[data][3]=='t' :
                        aux=obtain_data((file + filenames[data]),N)
                        WL.append(aux)
                
        complete_data=E
        complete_data=np.concatenate((complete_data,G),axis=0)
        complete_data=np.concatenate((complete_data,T),axis=0)
        complete_data=np.concatenate((complete_data,DS),axis=0)
        complete_data=np.concatenate((complete_data,DR),axis=0)
        complete_data=np.concatenate((complete_data,PH),axis=0)
        complete_data=np.concatenate((complete_data,PU),axis=0)
        complete_data=np.concatenate((complete_data,PO),axis=0)
        complete_data=np.concatenate((complete_data,SD),axis=0)
        complete_data=np.concatenate((complete_data,SI),axis=0)
        complete_data=np.concatenate((complete_data,SM),axis=0)
        complete_data=np.concatenate((complete_data,WD),axis=0)
        complete_data=np.concatenate((complete_data,WI),axis=0)
        complete_data=np.concatenate((complete_data,WL),axis=0)
        complete_data=np.concatenate((complete_data,WT),axis=0)


    
      
            
    return complete_data

def obtain_data(file_name,N):
    complete_secuence=np.load(file_name)
    for part in range(int(N/2)):
        row=[]
        j=0 #joint
        input_data=[]
        while j<15: 
            row=np.append(row,complete_secuence[part*2][j][0])
            row=np.append(row,complete_secuence[part*2][j][1])
            j+=1    
        input_data.append(row)
        row=[]   
        if part==0:
            D=input_data   
        else:
            D=np.concatenate((D,input_data),axis=0)
    return D


def load_data_qdrcn(typ,n_frames):
    print('Data loading')
    path='/home/chrisus/Proyectofinal/Motion/processed/'
    complete_data=[]
    test_data=[]
    cont=0
    if typ=='train':
        print('Data for training')
        ext=[1,5,6,7,8,9]
    if typ=='test':
        print('Data for testing')
        ext=[2]
    if typ=='val':
        print('Data for validation')
        ext=[11]
    for i in range(len(ext)):
        file=path + 'S' + str(ext[i])+'/'
        for (dirpath, dirnames, filenames)  in os.walk(file):        
            for data in range(len(filenames)):
                complete_secuence=np.load(file + filenames[data])
                N=len(complete_secuence[:])
                split=round(N/n_frames)
                for part in range(split):
                    row=[]
                    input_data=[]
                    for m in range(int(n_frames/2)):
                        ecu=m*2+part*n_frames
                        j=0 #joint
                        if part==(split-1) and (ecu)>=N:
                                ecu=2*N-ecu-2
                        while j<15:
                            row=np.append(row,complete_secuence[ecu][j][0])
                            row=np.append(row,complete_secuence[ecu][j][1])
                            j+=1
                        input_data.append(row)
                        row=[]
                    
                    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
                    complete_data.append(input_data)
                        
                #test_data.append(complete_data)
                if cont==0:
                    test_data=complete_data
                    complete_data=[]
                else:
                    test_data=np.concatenate((test_data,complete_data),axis=0)
                    complete_data=[]
                cont+=1
                #complete_data=[]
    return test_data
                    
def velocity(data):
    print('se realiza resta', np.shape(data))
    frames,_=np.shape(data)
    cont=0
    result=[]
    velo=[]
    for i in range (frames):
        if (i+1)<frames:
            subs=np.subtract(data[i+1],data[i])
            result.append(subs)
    velo.append(result)
    result=[]
    '''
        if sec==0:
            print(np.shape(result))
            velo=result
            result=[]
        else:
            velo=np.concatenate((velo,result),axis=0)
            result=[]'''
        
    return velo
                    
def process_for_norm(data,ns,fr,joints):
    h=[]
    t=[]
    p=[]
    rh=[]
    lh=[]
    rk=[]
    lk=[]
    ra=[]
    la=[]
    for i in range(ns):
        for y in range(fr):
            head=[]
            Torax=[]
            Pelvis=[]
            RHIP=[]
            LHIP=[]
            rkn=[]
            lkn=[]
            rak=[]
            lak=[]
            head=np.append(head,data[i][y][2])
            head=np.append(head,data[i][y][3])
            Pelvis=np.append(Pelvis,data[i][y][4])
            Pelvis=np.append(Pelvis,data[i][y][5])
            Torax=np.append(Torax,data[i][y][0])
            Torax=np.append(Torax,data[i][y][1])
            RHIP=np.append(RHIP,data[i][y][24])
            RHIP=np.append(RHIP,data[i][y][25])
            LHIP=np.append(LHIP,data[i][y][12])
            LHIP=np.append(LHIP,data[i][y][13])
            rkn=np.append(rkn,data[i][y][26])
            rkn=np.append(rkn,data[i][y][27])
            lkn=np.append(lkn,data[i][y][14])
            lkn=np.append(lkn,data[i][y][15])
            rak=np.append(rak,data[i][y][28])
            rak=np.append(rak,data[i][y][29])
            lak=np.append(lak,data[i][y][16])
            lak=np.append(lak,data[i][y][17])
            h.append(head)
            t.append(Torax)
            p.append(Pelvis)
            rh.append(RHIP)
            lh.append(LHIP)
            rk.append(rkn)
            lk.append(lkn)
            ra.append(rak)
            la.append(lak)

    return h,t,p,rh,lh,rk,lk,ra,la


def norm(data):
    ns,fr,joints=np.shape(data)
    h,t,p,rh,lh,rk,lk,ra,la=process_for_norm(data,ns,fr,joints)
    #lenght_head=distance.euclidean(t,h)
    lht_head=np.max(euclidean_dist(t,h))
    lht_torso=max(np.max(euclidean_dist(t,rh)),np.max(euclidean_dist(t,lh)),np.max(euclidean_dist(t,p)))
    lht_rleg=np.add(euclidean_dist(rh,rk),euclidean_dist(rk,ra))
    lht_lleg=np.add(euclidean_dist(lh,lk),euclidean_dist(lk,la))
    lht_legs=max(np.max(lht_lleg),np.max(lht_rleg))
    lht_body=lht_head+lht_torso+lht_legs
    norm_data=np.copy(data)
    for i in range(ns):
        for m in range(fr):
            cgx,cgy=center_of_gravity(data[i][m])
            for n in range(15):
                norm_data[i][m][n*2]=((data[i][m][n*2]-cgx)/lht_body)
                norm_data[i][m][n*2+1]=((data[i][m][n*2+1]-cgy)/lht_body)

    return norm_data

def center_of_gravity(data):
    joints=np.shape(data)
    joints=int(joints[0])
    jn=joints/2
    x=0
    y=0
    cx=0
    cy=0
    for n in range(int(jn)):
        x=x+data[n*2]
        y=y+data[n*2+1]
    cx=x/(joints/2)
    cy=y/(joints/2)
    return cx,cy


def euclidean_dist(a,b):
    ns,dim=np.shape(a)
    dist = (np.linalg.norm(np.subtract(a,b),axis=1)).reshape(ns,1)
    
    return dist



    
    





