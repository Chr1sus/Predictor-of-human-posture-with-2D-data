from __future__ import absolute_import
from __future__ import print_function
from pickle import TRUE
from selectors import PollSelector


#from translatev2 import train
'''Import useful packages'''
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import random
import math
import time
import h5py
from tqdm import tqdm
import wandb
'''Import torch'''
import torch
import torch.nn as nn
import torch.optim as opt
import torch
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms

'''Import tensorflow'''
import tensorflow as tf

'''Import load data for testing'''
import ops.loaddata as ld
from mat4py import loadmat
'''Import models'''
from Models.ATTN.ATTN import Encoder,EncoderLayer,Decoder,DecoderLayer, MultiHeadAttentionLayer, Seq2Seq
from Models.GCN.GCN_DCT import get_dct_matrix,AccumLoss,mpjpe_error_p3d,GCN,lr_decay
import Models.QDRNN.prediction_modelv2 as prediction_model
from Models.UNET.seg_data_ import SegmentationDataset
from Models.UNET.unet import UNet
import Models.UNET.config as config
import ops.vizu as vizu
from ops.utils import get_data,norm_matrix,transp,norm_from_spine, \
    norm_from_spine2,des_norm_matrix,des_transp,x_y_c,x_y_one,   \
    process_data_for_aberman,norm_m,calculate_error_per_frame

print('comparation')


FLAGS = tf.app.flags.FLAGS
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




def test_unet(input_dir,saved_dir):
    #saved_dir
    print('Load pretrained U-NET model')
    unetmodel = UNet().to(config.DEVICE)
    unetmodel.load_state_dict(torch.load(os.path.join('.../pretrained_models/unetv1.pth'))) #path for saved model
    unetmodel.eval()
    imagePaths = get_data(input_dir)
    print(np.shape(imagePaths))

    normin,max_now,min_now=norm_matrix(imagePaths)
    inpdata=transp(normin)
    inpdata=np.array(inpdata).astype(np.uint8)
    print(np.shape(inpdata[0:16]))
    transforms = transforms.Compose([transforms.ToPILImage(),
        transforms.Resize((config.INPUT_IMAGE_HEIGHT,
            config.INPUT_IMAGE_WIDTH)),
        transforms.ToTensor()])

    trainDS = SegmentationDataset(imagePaths=inpdata, maskPaths=inpdata,
        transforms=transforms)
    trainLoader = DataLoader(trainDS, shuffle=True,
        batch_size=1, pin_memory=config.PIN_MEMORY,
        num_workers=os.cpu_count())
    unettest=np.zeros((100,1,8,128))
    for (i, (x, y)) in enumerate(trainLoader):
            # send the input to the device
        (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))
        # perform a forward pass and calculate the training loss
        pred = unetmodel(x)
        y=pred.cpu().detach().numpy()

    unettest[i]=y
    print(np.shape(unettest))
    print(np.shape(np.transpose(unettest[0],(0,2,1))))
    x=des_transp(unettest)
    print(np.shape(x))
    z=des_norm_matrix(x,max_now,min_now)
    print(np.shape(z))
    np.save(saved_dir,z)

def test_gcn(input_dir,output_dir):
    #saved_dir
    print('Load pretrained GCN model')
    saved_dir=''
    #Define and load saved models
    gcnmodel = GCN(input_feature=64, hidden_feature=30, p_dropout=0.5,
                            num_stage=12, node_n=30)

    gcnmodel=torch.load(os.path.join('../pretrained_models/gcn_dct.pth'))

    print('Load ')
    gcnmodel.eval()

    inc_data=np.load(input_dir)
    test_penaction=np.zeros((1,64,15,2))
    test_penaction[0][:32]=inc_data
    #com_data,inc_data=ld.testing_data_1(test_path)
    #print(np.shape(com_data),end=' ')
    print('shape of input',np.shape(test_penaction))

    pose_input=ld.dataset_for_poses(test_penaction)
    pose_input=ld.norm(pose_input)
    n,frames,joints=np.shape(pose_input)
    dc,im=get_dct_matrix(64)
    gcntest=np.zeros((n,frames,joints))

    for m in range(n):
        input=np.matmul(dc,pose_input[m])
        input=np.transpose(input)
        input_t=torch.from_numpy(input)
        inputs = Variable(input_t).float()
        out=gcnmodel(inputs.cuda())
        pred=out.cpu().detach().numpy()
        ploting=np.matmul(pred,im)
        ploting=np.transpose(ploting)
        gcntest[m]=ploting
    print(np.shape(gcntest))
    
    np.save(output_dir,gcntest)



def creat_pen_data(save_dir_test,save_dir_ref,op,plus):
    pena=loadmat('path_of_pennaction_dataset')
    print(pena['nframes'], 'frames')
    xpen=pena['x']
    ypen=pena['y']
    skeleton=np.zeros((64,15,2))
    test_sk=np.zeros((32,15,2))


    for m in range(64):
        uv=m*plus+op
        skeleton[m][0][0]=(xpen[uv][1]+xpen[uv][2])/2
        skeleton[m][0][1]=(ypen[uv][1]+ypen[uv][2])/2

        skeleton[m][1][0]=xpen[uv][0]
        skeleton[m][1][1]=ypen[uv][0]

        skeleton[m][2][0]=(xpen[uv][7]+xpen[uv][8])/2
        skeleton[m][2][1]=(ypen[uv][7]+ypen[uv][8])/2

        for j in range(3,15,1):
            if j<9:
                skeleton[m][j][0]=xpen[uv][j*2-4]
                skeleton[m][j][1]=ypen[uv][j*2-4]
            else:
                skeleton[m][j][0]=xpen[uv][(j-9)*2+1]
                skeleton[m][j][1]=ypen[uv][(j-9)*2+1]


    test_sk=skeleton[:32]
    np.save(save_dir_test,test_sk)
    np.save(save_dir_ref,skeleton)

def att_test(input_data,saved_dir):
    INPUT_DIM=8
    OUTPUT_DIM=8
    HID_DIM = 8
    ENC_LAYERS = 3
    DEC_LAYERS = 3
    ENC_HEADS = 4
    DEC_HEADS = 4
    ENC_PF_DIM = 256
    DEC_PF_DIM = 256
    ENC_DROPOUT = 0.1
    DEC_DROPOUT = 0.1
    enc = Encoder(INPUT_DIM, 
                HID_DIM, 
                ENC_LAYERS, 
                ENC_HEADS, 
                ENC_PF_DIM, 
                ENC_DROPOUT, 
                device)

    dec = Decoder(OUTPUT_DIM, 
                HID_DIM, 
                DEC_LAYERS, 
                DEC_HEADS, 
                DEC_PF_DIM, 
                DEC_DROPOUT, 
                device)

    print('Load pretrained Attention model')
    attmodel = Seq2Seq(enc, dec, '0', '1', device).to(device)
    attmodel.load_state_dict(torch.load(os.path.join('../pretrained_models/bestseq2seqmodel.pt')))
    attmodel.eval()


    p32=input_data


    def pr(dir):
        x=[]
        for (dirpath, dirnames, filenames)  in os.walk(dir):
            for data in range(len(filenames)):
                    input=np.load(dir+filenames[data])
                    s,t,d=np.shape(input)
                    if d==4:
                        rest=np.zeros((128,8))
                        for n in range(t):
                            rest[n][:4]=input[0][n]
                        x.append(rest)
                    else:
                        x.append(input[0])

        print('datos en x',np.shape(x))
        return x


    in32=pr(p32)
    samples,_,_=np.shape(in32)
    fill=np.zeros((1,128,8))
    saved_test=np.zeros((samples,1,128,8))
    print('samples ',samples)
    for ns in range(samples):
        src = in32[ns]
        trg = fill
        src = torch.FloatTensor(src)
        trg = torch.FloatTensor(trg)
        output, _ = attmodel(src.cuda(),trg.cuda())
        out=output.cpu().detach().numpy()
        saved_test[ns]=out
  

    np.save(saved_dir,saved_test)





def vis_seq(p_targ,p_data,s,f):
    x_t,y_t=x_y_c(p_targ,s,f)
    x_d,y_d=x_y_c(p_data,s,f)
    sx0,sy0,lax0,lay0,lgx0,lgy0,rax0,ray0,rgx0,rgy0=vizu.creat_skeleton(x_t,y_t)
    sx1,sy1,lax1,lay1,lgx1,lgy1,rax1,ray1,rgx1,rgy1=vizu.creat_skeleton(x_d,y_d)

    print('plot of data')
    plt.figure(1)
    plt.plot(-sx0,-sy0, '--ko') 
    plt.plot(-lax0,-lay0, '--ko')
    plt.plot(-lgx0,-lgy0, '--ko')
    plt.plot(-rax0,-ray0, '--ko')
    plt.plot(-rgx0,-rgy0, '--ko')
    plt.plot(-sx1,-sy1, '-co') 
    plt.plot(-lax1,-lay1, '-ro')
    plt.plot(-lgx1,-lgy1, '-yo')
    plt.plot(-rax1,-ray1, '-bo')
    plt.plot(-rgx1,-rgy1, '-go')
    #plt.xlim([-0.08, -0.01])
    #plt.ylim([-0.08, -0.01])
    plt.show()
#prfinal

class Graph_result():
    def __init__(self):
        pass

    def normalize_one(self,data):
        samp,fram,_,_=np.shape(data)
        for n in range(samp):
            for fr in range(fram):
                data[n][fr]=np.interp(data[n][fr], (data[n][fr].min(), data[n][fr].max()), (-0.01, +0.01))
        return data

    def load_data(self,dir_targ,dir_pred):
        target=np.load(dir_targ)
        pred=np.load(dir_pred)
        target=process_data_for_aberman(ld.norm(ld.dataset_for_poses(target)))
        pred=process_data_for_aberman(ld.norm(ld.dataset_for_poses(pred)))

        ntarget,npred,_,_=norm_m(target,pred)
        return ntarget,npred

    def plot_four_methods(self,att,unet,target,qdrn,gcn):
        attr=np.load(att)
        unetr=np.load(unet)
        target=np.load(target)
        qdrn=np.load(qdrn)
        gcn0=np.load(gcn)
        attr=[attr]
        unetr=[unetr]
        attr=self.normalize_one(attr)
        unetr=self.normalize_one(unetr)
        target=self.normalize_one(target)
        qdrn=self.normalize_one(qdrn)
        gcn0=self.normalize_one(gcn0)


        fig, axs = plt.subplots(2, 2)
        fr=63
        print(np.shape(target[37][fr]))
        x_t,y_t,x_0,y_0,x_1,y_1,x_2,y_2,x_3,y_3=norm_from_spine(target[0][fr],attr[0][fr],unetr[0][fr],qdrn[0][fr],gcn0[0][fr])
        print(x_t)
        sx0,sy0,lax0,lay0,lgx0,lgy0,rax0,ray0,rgx0,rgy0=vizu.creat_skeleton(x_t,y_t)
        sx1,sy1,lax1,lay1,lgx1,lgy1,rax1,ray1,rgx1,rgy1=vizu.creat_skeleton(x_0,y_0)
        axs[0, 0].plot(-sx0,-sy0, '--ko') 
        axs[0, 0].plot(-lax0,-lay0, '--ko')
        axs[0, 0].plot(-lgx0,-lgy0, '--ko')
        axs[0, 0].plot(-rax0,-ray0, '--ko')
        axs[0, 0].plot(-rgx0,-rgy0, '--ko')
        axs[0, 0].plot(-sx1,-sy1, '-co') 
        axs[0, 0].plot(-lax1,-lay1, '-ro')
        axs[0, 0].plot(-lgx1,-lgy1, '-yo')
        axs[0, 0].plot(-rax1,-ray1, '-bo')
        axs[0, 0].plot(-rgx1,-rgy1, '-go')
        axs[0, 0].set_aspect('equal')
        axs[0, 0].set(xlim=(-0.15, 0.15), ylim=(-0.15, 0.15))
        axs[0, 0].set_title('Seq2seq con mecanismo de atención', fontsize=10)

        sx0,sy0,lax0,lay0,lgx0,lgy0,rax0,ray0,rgx0,rgy0=vizu.creat_skeleton(x_t,y_t)
        sx1,sy1,lax1,lay1,lgx1,lgy1,rax1,ray1,rgx1,rgy1=vizu.creat_skeleton(x_1,y_1)
        axs[0, 1].plot(-sx0,-sy0, '--ko') 
        axs[0, 1].plot(-lax0,-lay0, '--ko')
        axs[0, 1].plot(-lgx0,-lgy0, '--ko')
        axs[0, 1].plot(-rax0,-ray0, '--ko')
        axs[0, 1].plot(-rgx0,-rgy0, '--ko')
        axs[0, 1].plot(-sx1,-sy1, '-co') 
        axs[0, 1].plot(-lax1,-lay1, '-ro')
        axs[0, 1].plot(-lgx1,-lgy1, '-yo')
        axs[0, 1].plot(-rax1,-ray1, '-bo')
        axs[0, 1].plot(-rgx1,-rgy1, '-go')
        axs[0, 1].set_aspect('equal')
        axs[0, 1].set(xlim=(-0.15, 0.15), ylim=(-0.15, 0.15))
        axs[0, 1].set_title('U-NET', fontsize=10)

        sx0,sy0,lax0,lay0,lgx0,lgy0,rax0,ray0,rgx0,rgy0=vizu.creat_skeleton(x_t,y_t)
        sx1,sy1,lax1,lay1,lgx1,lgy1,rax1,ray1,rgx1,rgy1=vizu.creat_skeleton(x_2,y_2)
        axs[1, 0].plot(-sx0,-sy0, '--ko') 
        axs[1, 0].plot(-lax0,-lay0, '--ko')
        axs[1, 0].plot(-lgx0,-lgy0, '--ko')
        axs[1, 0].plot(-rax0,-ray0, '--ko')
        axs[1, 0].plot(-rgx0,-rgy0, '--ko')
        axs[1, 0].plot(-sx1,-sy1, '-co') 
        axs[1, 0].plot(-lax1,-lay1, '-ro')
        axs[1, 0].plot(-lgx1,-lgy1, '-yo')
        axs[1, 0].plot(-rax1,-ray1, '-bo')
        axs[1, 0].plot(-rgx1,-rgy1, '-go')
        axs[1, 0].set_aspect('equal')
        axs[1, 0].set(xlim=(-0.15, 0.15), ylim=(-0.15, 0.15))
        axs[1, 0].set_title('Q-DRNN', fontsize=10)


        sx0,sy0,lax0,lay0,lgx0,lgy0,rax0,ray0,rgx0,rgy0=vizu.creat_skeleton(x_t,y_t)
        sx1,sy1,lax1,lay1,lgx1,lgy1,rax1,ray1,rgx1,rgy1=vizu.creat_skeleton(x_3,y_3)
        axs[1, 1].plot(-sx0,-sy0, '--ko') 
        axs[1, 1].plot(-lax0,-lay0, '--ko')
        axs[1, 1].plot(-lgx0,-lgy0, '--ko')
        axs[1, 1].plot(-rax0,-ray0, '--ko')
        axs[1, 1].plot(-rgx0,-rgy0, '--ko')
        axs[1, 1].plot(-sx1,-sy1, '-co') 
        axs[1, 1].plot(-lax1,-lay1, '-ro')
        axs[1, 1].plot(-lgx1,-lgy1, '-yo')
        axs[1, 1].plot(-rax1,-ray1, '-bo')
        axs[1, 1].plot(-rgx1,-rgy1, '-go')
        axs[1, 1].set_aspect('equal')
        axs[1, 1].set(xlim=(-0.15, 0.15), ylim=(-0.15, 0.15))
        axs[1, 1].set_title('GCN con DCT', fontsize=10)

        fig.tight_layout()

        plt.show()

    def grafic_results_gcn_method(self,objt_pre,gcn_retrained,gcn_):
        bg1=np.load(gcn_retrained)
        final=np.load(objt_pre)
        final=self.normalize_one(final)
        bg1=self.normalize_one(bg1)
        targ=np.load(objt_pre)
        gcn1=np.load(gcn_)
        print('final shape',np.shape(gcn1))
        print('test_data_aberman shape',np.shape(targ))
        
        ngcn1,noutgcn1=self.load_data(self,targ,gcn1)


        '''
        vis_seq(noutgcn1,ngcn1,0,5)
        vis_seq(noutgcn1,ngcn1,1,63)
        vis_seq(noutgcn1,ngcn1,2,15)
        '''

        mngcn1,mnoutgcn1,_,_=norm_m(final,bg1)

        '''
        vis_seq(mngcn1,mnoutgcn1,93,0)
        vis_seq(mngcn1,mnoutgcn1,92,15)
        vis_seq(mngcn1,mnoutgcn1,94,30)
        '''

        MSEgen=calculate_error_per_frame(100,64,ngcn1,noutgcn1,None)
        MSEgcn=calculate_error_per_frame(100,64,mngcn1,mnoutgcn1,None)

        print('promedio total para error de 100 muestras',np.mean(MSEgcn))
        frames=list(range(0,64))
        plt.figure(1)
        plt.plot(frames,MSEgen, label='MSE for GCN on Aberman')
        plt.title("MSE para modelo seleccionado en el Aberman usando conjunto de datos de Pen Action")
        plt.xlabel("Frames")
        plt.ylabel("Mean MSE for 100 samples")
        plt.xlim(0,63)
        plt.legend()
        plt.show()



        plt.figure(2)
        plt.plot(frames,MSEgcn, label='MSE for GCN')           
        plt.title("MSE para modelo seleccionado en el Aberman usando conjunto de datos de Pen Action")
        plt.xlabel("Frames")
        plt.ylabel("Mean MSE para 100 muestras")
        plt.xlim(0,63)
        plt.legend()
        plt.show()

    def plot_best_method_trained_retrained(self,data_target,data_retrained,data_):

        

        #@final ploting
        target_aberman=np.load(data_target)
        trupred=np.load(data_retrained)
        bfpred=np.load(data_)
        print('trupred shape,',np.shape(trupred),'bf shape,', np.shape(bfpred))
        target_aberman=self.normalize_one(target_aberman)
        trupred=self.normalize_one(trupred)
        bfpred=self.normalize_one(bfpred)
       

        fin, ax = plt.subplots(2, 3)
        x_t,y_t,x_0,y_0=norm_from_spine2(target_aberman[0][16],trupred[0][16])
        sx0,sy0,lax0,lay0,lgx0,lgy0,rax0,ray0,rgx0,rgy0=vizu.creat_skeleton(x_t,y_t)
        sx1,sy1,lax1,lay1,lgx1,lgy1,rax1,ray1,rgx1,rgy1=vizu.creat_skeleton(x_0,y_0)
        ax[0, 0].plot(-sx0,-sy0, '--ko') 
        ax[0, 0].plot(-lax0,-lay0, '--ko')
        ax[0, 0].plot(-lgx0,-lgy0, '--ko')
        ax[0, 0].plot(-rax0,-ray0, '--ko')
        ax[0, 0].plot(-rgx0,-rgy0, '--ko')
        ax[0, 0].plot(-sx1,-sy1, '-co') 
        ax[0, 0].plot(-lax1,-lay1, '-ro')
        ax[0, 0].plot(-lgx1,-lgy1, '-yo')
        ax[0, 0].plot(-rax1,-ray1, '-bo')
        ax[0, 0].plot(-rgx1,-rgy1, '-go')
        ax[0, 0].set_aspect('equal')
        ax[0, 0].set(xlim=(-0.1, 0.1), ylim=(-0.12, 0.1))
        ax[0, 0].set_title('Cuadro 16 para GCN reentrenado', fontsize=10)

        x_t,y_t,x_0,y_0=norm_from_spine2(target_aberman[0][35],trupred[0][35])
        sx0,sy0,lax0,lay0,lgx0,lgy0,rax0,ray0,rgx0,rgy0=vizu.creat_skeleton(x_t,y_t)
        sx1,sy1,lax1,lay1,lgx1,lgy1,rax1,ray1,rgx1,rgy1=vizu.creat_skeleton(x_0,y_0)
        ax[0, 1].plot(-sx0,-sy0, '--ko') 
        ax[0, 1].plot(-lax0,-lay0, '--ko')
        ax[0, 1].plot(-lgx0,-lgy0, '--ko')
        ax[0, 1].plot(-rax0,-ray0, '--ko')
        ax[0, 1].plot(-rgx0,-rgy0, '--ko')
        ax[0, 1].plot(-sx1,-sy1, '-co') 
        ax[0, 1].plot(-lax1,-lay1, '-ro')
        ax[0, 1].plot(-lgx1,-lgy1, '-yo')
        ax[0, 1].plot(-rax1,-ray1, '-bo')
        ax[0, 1].plot(-rgx1,-rgy1, '-go')
        ax[0, 1].set_aspect('equal')
        ax[0, 1].set(xlim=(-0.1, 0.1), ylim=(-0.12, 0.1))
        ax[0, 1].set_title('Cuadro 35 para GCN reentrenado', fontsize=10)

        x_t,y_t,x_0,y_0=norm_from_spine2(target_aberman[0][55],trupred[0][55])
        sx0,sy0,lax0,lay0,lgx0,lgy0,rax0,ray0,rgx0,rgy0=vizu.creat_skeleton(x_t,y_t)
        sx1,sy1,lax1,lay1,lgx1,lgy1,rax1,ray1,rgx1,rgy1=vizu.creat_skeleton(x_0,y_0)
        ax[0, 2].plot(-sx0,-sy0, '--ko') 
        ax[0, 2].plot(-lax0,-lay0, '--ko')
        ax[0, 2].plot(-lgx0,-lgy0, '--ko')
        ax[0, 2].plot(-rax0,-ray0, '--ko')
        ax[0, 2].plot(-rgx0,-rgy0, '--ko')
        ax[0, 2].plot(-sx1,-sy1, '-co') 
        ax[0, 2].plot(-lax1,-lay1, '-ro')
        ax[0, 2].plot(-lgx1,-lgy1, '-yo')
        ax[0, 2].plot(-rax1,-ray1, '-bo')
        ax[0, 2].plot(-rgx1,-rgy1, '-go')
        ax[0, 2].set_aspect('equal')
        ax[0, 2].set(xlim=(-0.1, 0.1), ylim=(-0.12, 0.1))
        ax[0, 2].set_title('Cuadro 55 para GCN reentrenado', fontsize=10)

        x_t,y_t,x_0,y_0=norm_from_spine2(target_aberman[0][16],bfpred[0][16])
        sx0,sy0,lax0,lay0,lgx0,lgy0,rax0,ray0,rgx0,rgy0=vizu.creat_skeleton(x_t,y_t)
        sx1,sy1,lax1,lay1,lgx1,lgy1,rax1,ray1,rgx1,rgy1=vizu.creat_skeleton(x_0,y_0)
        ax[1, 0].plot(-sx0,-sy0, '--ko') 
        ax[1, 0].plot(-lax0,-lay0, '--ko')
        ax[1, 0].plot(-lgx0,-lgy0, '--ko')
        ax[1, 0].plot(-rax0,-ray0, '--ko')
        ax[1, 0].plot(-rgx0,-rgy0, '--ko')
        ax[1, 0].plot(-sx1,-sy1, '-co') 
        ax[1, 0].plot(-lax1,-lay1, '-ro')
        ax[1, 0].plot(-lgx1,-lgy1, '-yo')
        ax[1, 0].plot(-rax1,-ray1, '-bo')
        ax[1, 0].plot(-rgx1,-rgy1, '-go')
        ax[1, 0].set_aspect('equal')
        ax[1, 0].set(xlim=(-0.1, 0.1), ylim=(-0.12, 0.1))
        ax[1, 0].set_title('Cuadro 16 para GCN previo a reentrenamiento', fontsize=10)

        x_t,y_t,x_0,y_0=norm_from_spine2(target_aberman[0][35],bfpred[0][35])
        sx0,sy0,lax0,lay0,lgx0,lgy0,rax0,ray0,rgx0,rgy0=vizu.creat_skeleton(x_t,y_t)
        sx1,sy1,lax1,lay1,lgx1,lgy1,rax1,ray1,rgx1,rgy1=vizu.creat_skeleton(x_0,y_0)
        ax[1, 1].plot(-sx0,-sy0, '--ko') 
        ax[1, 1].plot(-lax0,-lay0, '--ko')
        ax[1, 1].plot(-lgx0,-lgy0, '--ko')
        ax[1, 1].plot(-rax0,-ray0, '--ko')
        ax[1, 1].plot(-rgx0,-rgy0, '--ko')
        ax[1, 1].plot(-sx1,-sy1, '-co') 
        ax[1, 1].plot(-lax1,-lay1, '-ro')
        ax[1, 1].plot(-lgx1,-lgy1, '-yo')
        ax[1, 1].plot(-rax1,-ray1, '-bo')
        ax[1, 1].plot(-rgx1,-rgy1, '-go')
        ax[1, 1].set_aspect('equal')
        ax[1, 1].set(xlim=(-0.1, 0.1), ylim=(-0.12, 0.1))
        ax[1, 1].set_title('Cuadro 35 para GCN previo a reentrenamiento', fontsize=10)

        x_t,y_t,x_0,y_0=norm_from_spine2(target_aberman[0][55],bfpred[0][55])
        sx0,sy0,lax0,lay0,lgx0,lgy0,rax0,ray0,rgx0,rgy0=vizu.creat_skeleton(x_t,y_t)
        sx1,sy1,lax1,lay1,lgx1,lgy1,rax1,ray1,rgx1,rgy1=vizu.creat_skeleton(x_0,y_0)
        ax[1, 2].plot(-sx0,-sy0, '--ko') 
        ax[1, 2].plot(-lax0,-lay0, '--ko')
        ax[1, 2].plot(-lgx0,-lgy0, '--ko')
        ax[1, 2].plot(-rax0,-ray0, '--ko')
        ax[1, 2].plot(-rgx0,-rgy0, '--ko')
        ax[1, 2].plot(-sx1,-sy1, '-co') 
        ax[1, 2].plot(-lax1,-lay1, '-ro')
        ax[1, 2].plot(-lgx1,-lgy1, '-yo')
        ax[1, 2].plot(-rax1,-ray1, '-bo')
        ax[1, 2].plot(-rgx1,-rgy1, '-go')
        ax[1, 2].set_aspect('equal')
        ax[1, 2].set(xlim=(-0.1, 0.1), ylim=(-0.12, 0.1))
        ax[1, 2].set_title('Cuadro 55 para GCN previo a reentrenamiento', fontsize=10)

        fin.tight_layout()

        plt.show()


        MSEg0=calculate_error_per_frame(1,64,trupred,target_aberman,None)
        MSEg1=calculate_error_per_frame(1,64,bfpred,target_aberman,None)

        print('promedio total para error de gcn reentrenado',np.mean(MSEg0))
        print('promedio total para error de gcn  no reentrenado',np.mean(MSEg1))
        frames=list(range(0,64))
        plt.figure(3)
        plt.plot(frames,MSEg0, label='MSE para GCN reentrenado con Panoptics')
        plt.plot(frames,MSEg1, label='MSE para GCN')
        plt.title("MSE para modelo seleccionado en Aberman usando el conjunto de datos de Pen Action")
        plt.xlabel("Cuadros")
        plt.ylabel("MSE promedio")
        plt.xlim(0,63)
        plt.legend()
        plt.show()


#@For qdrn data test and output generation
def test_qdrnn(input_dir,saved_dir):
    FLAGS = tf.app.flags.FLAGS
    print('Load pretrained tensorflow model')
    with tf.Session() as sess:

        model = prediction_model.Seq2SeqModel(
        FLAGS.seq_length_in,
        FLAGS.seq_length_out,
        FLAGS.size, # hidden layer size
        FLAGS.num_layers,
        FLAGS.max_gradient_norm,
        FLAGS.batch_size,
        FLAGS.learning_rate,
        FLAGS.learning_rate_decay_factor,
        'summary',
        128,
        FLAGS.max_diffusion_step,
        FLAGS.filter_type,
        not FLAGS.omit_one_hot,
        FLAGS.eval_pose,
        dtype=tf.float32)

        #ckpt = tf.train.get_checkpoint_state( os.path.join(saved_dir,'Q-DRCN/'), latest_filename="checkpoint")
        #ckpt_name=os.path.normpath(os.path.join( os.path.join(os.path.normpath("./output/Q-DRCN/"),"checkpoint")))
        #print(ckpt)
        print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        #print(ckpt_name)
        #ckptname='/home/chrisus/Proyectofinal/GIT/Modelos/experiments/all/out_32/iterations_5000/omit_one_hot/size_30/lr_0.0001/checkpoint5000.meta'
        #model.saver.restore( sess, ckptname )
        load= tf.train.import_meta_graph(os.path.join('../pretrained_models/Q-DRNN_MODEL/complete.meta'))
        load.restore( sess, os.path.join('../pretrained_models/Q-DRNN_MODELQ-DRCN/complete'))
        print('Loaded')
        print(model)
        #for prediction
        inc_data=np.load(input_dir)
        pose_input=ld.dataset_for_poses(inc_data)
        pose_input=ld.norm(pose_input)
        train_set=ld.norm(pose_input)
        ns,fr,jn=np.shape(train_set)
        #actions=list(range(0,1))
        #train_set=dict(zip(actions,train_set))
        batch_size=8
        source_seq_len=32
        input_size_target=30
        target_seq_len=32
        total_frames=64
        encoder_inputs  = np.zeros((batch_size, source_seq_len-2, input_size_target), dtype=float)
        decoder_inputs  = np.zeros((batch_size, target_seq_len, input_size_target), dtype=float)
        decoder_outputs = np.zeros((batch_size, target_seq_len, input_size_target), dtype=float)
        all_poses = np.zeros((batch_size, total_frames, input_size_target), dtype=float)
        op=11
        for m in range(batch_size):
            data_sel=train_set[m+batch_size*op][:]

            encoder_inputs[m,:,0:input_size_target] = data_sel[1:source_seq_len-1, :] - data_sel[0:source_seq_len-2, :]
            decoder_inputs[m,:,0:input_size_target] = data_sel[source_seq_len-1:source_seq_len+target_seq_len-1, :] - data_sel[source_seq_len-2:source_seq_len+target_seq_len-2, :]
            decoder_outputs[m,:,0:input_size_target] = data_sel[source_seq_len:, 0:input_size_target] - data_sel[source_seq_len-1:-1, 0:input_size_target]
            all_poses[m, :, 0:input_size_target] = data_sel[:, 0:input_size_target]
            #action_prefix, action_postfix_input, action_postfix_output, action_poses = model.get_batch(train_set,actions)

        predc1, predc2,prediction_qdrc = model.step(sess, encoder_inputs, decoder_inputs, decoder_outputs, all_poses, forward_only=False,srnn_seeds=True)


        y=np.transpose(prediction_qdrc, (1, 0, 2))
        train_set[8*op][32:]=y[0]
        train_set[(8*op+1)][32:]=y[1]
        train_set[(8*op+2)][32:]=y[2]
        train_set[(8*op+3)][32:]=y[3]
        train_set[(8*op+4)][32:]=y[4]
        train_set[(8*op+5)][32:]=y[5]
        train_set[(8*op+6)][32:]=y[6]
        train_set[(8*op+7)][32:]=y[7]
        data_sel=train_set[(op*8):(op*8+8)]
        print(np.shape(data_sel))
        np.save(saved_dir,data_sel)



#Generate the output for QDRN and GCN methods
outgcn='dir for output data.npy'
training_data='dir for input data'
test_gcn(training_data,outgcn)

outqdrnn='dir for output data.npy'
test_qdrnn(training_data,outqdrnn)

##Generate the output for UNET and Seq2Seq methods
out_unet_='dir for output data.npy'
tr_coded_data='dir for input coded data generated on Aberman'
test_unet(tr_coded_data,out_unet_)

out_att_='dir for output data.npy'
att_test(tr_coded_data,out_att_)


##After generating the outputs for the two methods using Aberman.
##create graphs to see results using the class Graph_result







