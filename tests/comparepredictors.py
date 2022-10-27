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
#import tensorflow as tf

'''Import load data for testing'''
import loaddata as ld
from mat4py import loadmat
'''Import models'''
from ATTN import Encoder,EncoderLayer,Decoder,DecoderLayer, MultiHeadAttentionLayer, Seq2Seq
from GCN_DCT import get_dct_matrix,AccumLoss,mpjpe_error_p3d,GCN,lr_decay
#import prediction_modelv2 as prediction_model
from seg_data_ import SegmentationDataset
from unet import UNet
import config
import vizu


print('comparation')


#FLAGS = tf.app.flags.FLAGS
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




def test_unet(saved_dir):
    #saved_dir
    print('Load pretrained U-NET model')
    saved_dir='/home/chrisus/Proyectofinal/GIT/Modelos/output/'
    unetmodel = UNet().to(config.DEVICE)
    unetmodel.load_state_dict(torch.load(os.path.join(saved_dir,'unetmodel/unetv1.pth')))
    unetmodel.eval()
    imagePaths = get_data('/home/chrisus/Proyectofinal/Motion/disptec-2020/tools/motion_predictor/train/')
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
    np.save('/home/chrisus/Proyectofinal/Motion/disptec-2020/tools/motion_predictor/output/unetinput.npy',z)

def test_gcn(input_dir,output_dir):
    #saved_dir
    print('Load pretrained GCN model')
    saved_dir='/home/chrisus/Proyectofinal/GIT/Modelos/output/'
    #Define and load saved models
    gcnmodel = GCN(input_feature=64, hidden_feature=30, p_dropout=0.5,
                            num_stage=12, node_n=30)

    gcnmodel=torch.load(os.path.join(saved_dir,'GCN_DCT/gcn_dctv3.pth'))

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
    pena=loadmat('/home/chrisus/Desktop/Penn_Action/labels/0001.mat')
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
#op=21
#save_dir='/home/chrisus/Proyectofinal/Motion/disptec-2020/tools/motion_predictor/output/final/testing/input_pred00.npy'
#save_dir_ref='/home/chrisus/Proyectofinal/Motion/disptec-2020/tools/motion_predictor/output/final/testing/input_aber00.npy'
#plus=2
#creat_pen_data(save_dir,save_dir_ref,op,plus)
#save_dir='/home/chrisus/Proyectofinal/Motion/disptec-2020/tools/motion_predictor/output/final/testing/input_pred01.npy'
#save_dir_ref='/home/chrisus/Proyectofinal/Motion/disptec-2020/tools/motion_predictor/output/final/testing/input_aber01.npy'
#creat_pen_data(save_dir,save_dir_ref,op,1)

#input_dir='/home/chrisus/Proyectofinal/Motion/disptec-2020/tools/motion_predictor/output/final/testing/input_pred00.npy'
#output_dir='/home/chrisus/Proyectofinal/Motion/disptec-2020/tools/motion_predictor/output/final/testing/bfpred00.npy'
#test_gcn(input_dir,output_dir)
#input_dir='/home/chrisus/Proyectofinal/Motion/disptec-2020/tools/motion_predictor/output/final/input_pred01.npy'
#output_dir='/home/chrisus/Proyectofinal/Motion/disptec-2020/tools/motion_predictor/output/final/trupred01.npy'
#test_gcn(input_dir,output_dir)
#input_dir='/home/chrisus/Proyectofinal/Motion/disptec-2020/tools/motion_predictor/output/final/input_pred02.npy'
#output_dir='/home/chrisus/Proyectofinal/Motion/disptec-2020/tools/motion_predictor/output/final/trupred02.npy'
#test_gcn(input_dir,output_dir)  
''''''   
def att_test(saved_dir):
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
    #saved_dir='/home/chrisus/Proyectofinal/GIT/Modelos/output/'
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
    attmodel.load_state_dict(torch.load(os.path.join(saved_dir,'ATTmodel/bestseq2seqmodel.pt')))
    attmodel.eval()


    p32='/home/chrisus/Proyectofinal/Motion/disptec-2020/tools/motion_predictor/train/'


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
    saved_test=np.zeros((100,1,128,8))
    print('samples ',samples)
    for ns in range(samples):
        src = in32[ns]
        trg = fill
        src = torch.FloatTensor(src)
        trg = torch.FloatTensor(trg)
        output, _ = attmodel(src.cuda(),trg.cuda())
        out=output.cpu().detach().numpy()
        saved_test[ns]=out
  

    #np.save('/home/chrisus/Proyectofinal/Motion/disptec-2020/tools/motion_predictor/output/attinput.npy',saved_test)





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
bg1=np.load('/home/chrisus/Proyectofinal/Motion/disptec-2020/tools/motion_predictor/output/prfinal.npy')
final=np.load('/home/chrisus/Proyectofinal/Motion/disptec-2020/tools/motion_predictor/output/test_data_aberman.npy')
print('target_test shape',np.shape(final))
print('test_data shape',np.shape(bg1))
targ=np.load('/home/chrisus/Proyectofinal/Motion/disptec-2020/tools/motion_predictor/output/target_test.npy')
gcn1=np.load('/home/chrisus/Proyectofinal/Motion/disptec-2020/tools/motion_predictor/output/final2.npy')
print('final shape',np.shape(gcn1))
print('test_data_aberman shape',np.shape(targ))


gcn1=process_data_for_aberman(ld.norm(ld.dataset_for_poses(gcn1)))
targ=process_data_for_aberman(ld.norm(ld.dataset_for_poses(targ)))
ngcn1,noutgcn1,_,_=norm_m(targ,gcn1)


'''
vis_seq(noutgcn1,ngcn1,0,5)
vis_seq(noutgcn1,ngcn1,1,63)
vis_seq(noutgcn1,ngcn1,2,15)
vis_seq(noutgcn1,ngcn1,3,62)
vis_seq(noutgcn1,ngcn1,4,5)
vis_seq(noutgcn1,ngcn1,5,63)
vis_seq(noutgcn1,ngcn1,6,15)
vis_seq(noutgcn1,ngcn1,7,62)
vis_seq(noutgcn1,ngcn1,8,63)
vis_seq(noutgcn1,ngcn1,9,15)
vis_seq(noutgcn1,ngcn1,10,62)
vis_seq(noutgcn1,ngcn1,11,63)
vis_seq(noutgcn1,ngcn1,12,15)
vis_seq(noutgcn1,ngcn1,13,62)
'''
samp,fram,_,_=np.shape(final)
for n in range(samp):
    for fr in range(fram):
        final[n][fr]=np.interp(final[n][fr], (final[n][fr].min(), final[n][fr].max()), (-0.01, +0.01))

samp,fram,_,_=np.shape(bg1)
for n in range(samp):
    for fr in range(fram):
        bg1[n][fr]=np.interp(bg1[n][fr], (bg1[n][fr].min(), bg1[n][fr].max()), (-0.01, 0.01))
#com2=process_data_for_aberman(ld.norm(ld.dataset_for_poses(final)))
#bg1=process_data_for_aberman(ld.norm(ld.dataset_for_poses(bg1)))
mngcn1,mnoutgcn1,_,_=norm_m(final,bg1)

'''
vis_seq(mngcn1,mnoutgcn1,93,0)
vis_seq(mngcn1,mnoutgcn1,92,15)
vis_seq(mngcn1,mnoutgcn1,94,30)
vis_seq(mngcn1,mnoutgcn1,95,36)
vis_seq(mngcn1,mnoutgcn1,96,47)
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

prevfinal=np.load('/home/chrisus/Proyectofinal/Motion/disptec-2020/tools/motion_predictor/output/final.npy')
testingretargeting=np.load('/home/chrisus/Proyectofinal/Motion/disptec-2020/tools/motion_predictor/output/target_testNO.npy')
prevfinal=process_data_for_aberman(ld.norm(ld.dataset_for_poses(prevfinal)))
testingretargeting=process_data_for_aberman(ld.norm(ld.dataset_for_poses(testingretargeting)))

ng,nout,_,_=norm_m(testingretargeting,prevfinal)
#vis_seq(-ng,-nout,5,45)
#vis_seq(-ng,-nout,6,10)
#vis_seq(-ng,-nout,7,45)
#vis_seq(-ng,-nout,8,10)
#Define test data

#np.save('/home/chrisus/Proyectofinal/Motion/disptec-2020/tools/motion_predictor/output/gcninput.npy',gcntest)
#np.save('/home/chrisus/Proyectofinal/Motion/disptec-2020/tools/motion_predictor/output/inc_data.npy',inc_data)
#np.save('/home/chrisus/Proyectofinal/Motion/disptec-2020/tools/motion_predictor/output/com_data.npy',com_data)
#attr=np.load('/home/chrisus/Proyectofinal/Motion/disptec-2020/tools/motion_predictor/output/attresult.npy')
attr=np.load('/home/chrisus/Desktop/Panoptics testing/out_att_aberman.npy')
#unetr=np.load('/home/chrisus/Proyectofinal/Motion/disptec-2020/tools/motion_predictor/output/unetresult.npy')
unetr=np.load('/home/chrisus/Desktop/Panoptics testing/out_unet_aberman.npy')
target=np.load('/home/chrisus/Proyectofinal/Motion/disptec-2020/tools/motion_predictor/output/target.npy')
qdrn=np.load('/home/chrisus/Proyectofinal/Motion/disptec-2020/tools/motion_predictor/output/qdrnfinal.npy')
gcn0=np.load('/home/chrisus/Proyectofinal/Motion/disptec-2020/tools/motion_predictor/output/gcnconfinal.npy')
attr=[attr]
unetr=[unetr]
samp,fram,_,_=np.shape(attr)
for n in range(samp):
    for fr in range(fram):
        attr[n][fr]=np.interp(attr[n][fr], (attr[n][fr].min(), attr[n][fr].max()), (-0.1, +0.1))

samp,fram,_,_=np.shape(unetr)
for n in range(samp):
    for fr in range(fram):
        unetr[n][fr]=np.interp(unetr[n][fr], (unetr[n][fr].min(), unetr[n][fr].max()), (-0.1, +0.1))


samp,fram,_,_=np.shape(target)
for n in range(samp):
    for fr in range(fram):
        target[n][fr]=np.interp(target[n][fr], (target[n][fr].min(), target[n][fr].max()), (-0.1, +0.1))

samp,fram,_,_=np.shape(qdrn)
for n in range(samp):
    for fr in range(fram):
        qdrn[n][fr]=np.interp(qdrn[n][fr], (qdrn[n][fr].min(), qdrn[n][fr].max()), (-0.1, +0.1))

samp,fram,_,_=np.shape(gcn0)
for n in range(samp):
    for fr in range(fram):
        gcn0[n][fr]=np.interp(gcn0[n][fr], (gcn0[n][fr].min(), gcn0[n][fr].max()), (-0.1, +0.1))





#attr=process_data_for_aberman(ld.norm(ld.dataset_for_poses(attr)))
#unetr=process_data_for_aberman(ld.norm(ld.dataset_for_poses(unetr)))
#target=process_data_for_aberman(ld.norm(ld.dataset_for_poses(target)))
#qdrn=process_data_for_aberman(ld.norm(ld.dataset_for_poses(qdrn)))
#gcn0=process_data_for_aberman(ld.norm(ld.dataset_for_poses(gcn0)))

fig, axs = plt.subplots(2, 2)
fr=63
print(np.shape(target[37][fr]))
x_t,y_t,x_0,y_0,x_1,y_1,x_2,y_2,x_3,y_3=norm_from_spine(target[0][fr],attr[0][fr],unetr[0][fr],qdrn[0][fr],gcn0[0][fr])
#attr,target,_,_=norm_m(attr,target)
#x_t,y_t=x_y_c(target,0,2)
#x_d,y_d=x_y_c(attr,0,2)
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

#unetr,target,_,_=norm_m(unetr,target)
#x_t,y_t=x_y_c(target,0,2)
#x_d,y_d=x_y_c(unetr,0,2)
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

#qdrn,target,_,_=norm_m(qdrn,target)
#x_t,y_t=x_y_c(target,0,37)
#x_d,y_d=x_y_c(qdrn,0,37)
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

#gcn0,target,_,_=norm_m(gcn0,target)
#x_t,y_t=x_y_c(target,0,37)
#x_d,y_d=x_y_c(gcn0,0,37)
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

#@final ploting
target_aberman=np.load('/home/chrisus/Proyectofinal/Motion/disptec-2020/tools/motion_predictor/output/final/testing/target_final_gt00.npy')
trupred=np.load('/home/chrisus/Proyectofinal/Motion/disptec-2020/tools/motion_predictor/output/final/testing/trufinal11.npy')
bfpred=np.load('/home/chrisus/Proyectofinal/Motion/disptec-2020/tools/motion_predictor/output/final/testing/bffinal11.npy')
print('trupred shape,',np.shape(trupred),'bf shape,', np.shape(bfpred))
samp,fram,_,_=np.shape(target_aberman)
for n in range(samp):
    for fr in range(fram):
        target_aberman[n][fr]=np.interp(target_aberman[n][fr], (target_aberman[n][fr].min(), target_aberman[n][fr].max()), (-0.1, +0.1))

samp,fram,_,_=np.shape(trupred)
for n in range(samp):
    for fr in range(fram):
        trupred[n][fr]=np.interp(trupred[n][fr], (trupred[n][fr].min(), trupred[n][fr].max()), (-0.1, +0.1))


samp,fram,_,_=np.shape(bfpred)
for n in range(samp):
    for fr in range(fram):
        bfpred[n][fr]=np.interp(bfpred[n][fr], (bfpred[n][fr].min(), bfpred[n][fr].max()), (-0.1, +0.1))


#for 01
target_aberman01=np.load('/home/chrisus/Proyectofinal/Motion/disptec-2020/tools/motion_predictor/output/final/testing/target_final_gt00.npy')
trupred01=np.load('/home/chrisus/Proyectofinal/Motion/disptec-2020/tools/motion_predictor/output/final/testing/trufinal11.npy')
bfpred01=np.load('/home/chrisus/Proyectofinal/Motion/disptec-2020/tools/motion_predictor/output/final/testing/bffinal11.npy')
print('trupred shape 01,',np.shape(trupred01),'bf shape 01,', np.shape(bfpred01))
samp,fram,_,_=np.shape(target_aberman01)
for n in range(samp):
    for fr in range(fram):
        target_aberman01[n][fr]=np.interp(target_aberman01[n][fr], (target_aberman01[n][fr].min(), target_aberman01[n][fr].max()), (-0.1, +0.1))

samp,fram,_,_=np.shape(trupred01)
for n in range(samp):
    for fr in range(fram):
        trupred01[n][fr]=np.interp(trupred01[n][fr], (trupred01[n][fr].min(), trupred01[n][fr].max()), (-0.1, +0.1))


samp,fram,_,_=np.shape(bfpred01)
for n in range(samp):
    for fr in range(fram):
        bfpred01[n][fr]=np.interp(bfpred01[n][fr], (bfpred01[n][fr].min(), bfpred01[n][fr].max()), (-0.1, +0.1))

#for 02
target_aberman02=np.load('/home/chrisus/Proyectofinal/Motion/disptec-2020/tools/motion_predictor/output/final/testing/target_final_gt00.npy')
trupred02=np.load('/home/chrisus/Proyectofinal/Motion/disptec-2020/tools/motion_predictor/output/final/testing/trufinal11.npy')
bfpred02=np.load('/home/chrisus/Proyectofinal/Motion/disptec-2020/tools/motion_predictor/output/final/testing/bffinal11.npy')
print('trupred shape 02,',np.shape(trupred02),'bf shape 02,', np.shape(bfpred02))
samp,fram,_,_=np.shape(target_aberman02)
for n in range(samp):
    for fr in range(fram):
        target_aberman02[n][fr]=np.interp(target_aberman02[n][fr], (target_aberman02[n][fr].min(), target_aberman02[n][fr].max()), (-0.1, +0.1))

samp,fram,_,_=np.shape(trupred02)
for n in range(samp):
    for fr in range(fram):
        trupred02[n][fr]=np.interp(trupred02[n][fr], (trupred02[n][fr].min(), trupred02[n][fr].max()), (-0.1, +0.1))


samp,fram,_,_=np.shape(bfpred02)
for n in range(samp):
    for fr in range(fram):
        bfpred02[n][fr]=np.interp(bfpred02[n][fr], (bfpred02[n][fr].min(), bfpred02[n][fr].max()), (-0.1, +0.1))


#trupred=process_data_for_aberman(trupred)
#bfpred=process_data_for_aberman(bfpred)



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

x_t,y_t,x_0,y_0=norm_from_spine2(target_aberman01[0][35],trupred01[0][35])
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

x_t,y_t,x_0,y_0=norm_from_spine2(target_aberman02[0][55],trupred02[0][55])
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

x_t,y_t,x_0,y_0=norm_from_spine2(target_aberman01[0][35],bfpred01[0][35])
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

x_t,y_t,x_0,y_0=norm_from_spine2(target_aberman02[0][55],bfpred02[0][55])
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


'''
def test_qdrnn(saved_dir):
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
        load= tf.train.import_meta_graph(os.path.join(saved_dir,'Q-DRCN/complete.meta'))
        load.restore( sess, os.path.join(saved_dir,'Q-DRCN/complete'))
        print('Loaded')
        print(model)
        #for prediction
        inc_data=np.load('/home/chrisus/Proyectofinal/Motion/disptec-2020/tools/motion_predictor/output/test_data_.npy')
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
        np.save('/home/chrisus/Proyectofinal/Motion/disptec-2020/tools/motion_predictor/output/bacth12qdrc.npy',data_sel)
        #print(np.shape(p1))
        #print(np.shape(p1))

'''


