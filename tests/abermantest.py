from itertools import count
import numpy as np
import os,sys
import torch
from ATTN import Encoder,EncoderLayer,Decoder,DecoderLayer, MultiHeadAttentionLayer, Seq2Seq
from unet import UNet
import config
from seg_data_ import SegmentationDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import vizu
import imageio
from PIL import Image,ImageFont, ImageDraw 
import glob  #use it if you want to read all of the certain file type in the directory

input_data_target=np.load('/home/chrisus/Proyectofinal/Motion/disptec-2020/tools/motion_predictor/output/final/testing/input_aberman_test_for_att_unet_.npy')
print(input_data_target)
print('shape',np.shape(input_data_target))
print('max value',np.max(input_data_target),'min value',np.min(input_data_target))
print('max value abs',np.max(np.absolute(input_data_target)),'min value',np.min(np.absolute(input_data_target)))


def att_test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    saved_dir='/home/chrisus/Proyectofinal/GIT/Modelos/output/'
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


    p32='/home/chrisus/Proyectofinal/Motion/disptec-2020/tools/motion_predictor/output/final/testing/input_aberman_test_for_att_unet_.npy'
    in32=np.load(p32)
    rest=np.zeros((1,128,8))
    for n in range(128):
        rest[0][n][:4]=in32[0][n]


    fill=np.zeros((1,128,8))

    
    src = rest[0]
    trg = fill
    src = torch.FloatTensor(src)
    trg = torch.FloatTensor(trg)
    output, _ = attmodel(src.cuda(),trg.cuda())
    out=output.cpu().detach().numpy()

    print(np.shape(out),'shape of output')
    print(out)
    print('max value',np.max(out),'min value',np.min(out))
    print('max value abs',np.max(np.absolute(out)),'min value',np.min(np.absolute(out)))
    out[0]=np.interp(out[0], (out[0].min(), out[0].max()), (-1.0046785, +2.5713344))
    print(out)
    np.save('/home/chrisus/Desktop/Panoptics testing/att_test.npy',out)

def get_data(path):
    input=np.load(path)
    info = []
    w=[]
    print('reading encoded data')

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

   
def test_unet():
    #saved_dir
    print('Load pretrained U-NET model')
    saved_dir='/home/chrisus/Proyectofinal/GIT/Modelos/output/'
    unetmodel = UNet().to(config.DEVICE)
    unetmodel.load_state_dict(torch.load(os.path.join(saved_dir,'unetmodel/unetv1.pth')))
    unetmodel.eval()
    imagePaths = get_data('/home/chrisus/Proyectofinal/Motion/disptec-2020/tools/motion_predictor/output/final/testing/input_aberman_test_for_att_unet_.npy')
    print(np.shape(imagePaths),'input data for unet shape')

    normin,max_now,min_now=norm_matrix(imagePaths)
    inpdata=transp(normin)
    inpdata=np.array(inpdata).astype(np.uint8)
    print(np.shape(inpdata[0:16]))
    transform = transforms.Compose([transforms.ToPILImage(),
        transforms.Resize((config.INPUT_IMAGE_HEIGHT,
            config.INPUT_IMAGE_WIDTH)),
        transforms.ToTensor()])

    trainDS = SegmentationDataset(imagePaths=inpdata, maskPaths=inpdata,
        transforms=transform)
    trainLoader = DataLoader(trainDS, shuffle=True,
        batch_size=1, pin_memory=config.PIN_MEMORY,
        num_workers=os.cpu_count())
    unettest=np.zeros((1,1,8,128))
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

    print(np.shape(z),'shape of output')
    print(z)
    print('max value',np.max(z),'min value',np.min(z))
    print('max value abs',np.max(np.absolute(z)),'min value',np.min(np.absolute(z)))
    z[0]=np.interp(z[0], (z[0].min(), z[0].max()), (-1.0046785, +2.5713344))
    print(z)
    np.save('/home/chrisus/Desktop/Panoptics testing/unet_test.npy',z)

#att_test()
#test_unet()

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


def x_y_one(data):

    n=0
    x=[]
    y=[]

    while n<15:
        x=np.append(x,data[n][0])
        y=np.append(y,data[n][1])
   
        n+=1

    return x,y

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

'''
fig, (ax1, ax2) = plt.subplots(2, 1)
fig.subplots_adjust(hspace=0.5)
fr = input("Please enter a string:\n")
print(f'You entered {fr}') 
fr = int(fr)

x_t,y_t,x_0,y_0=norm_from_spine2(target_aberman[0][fr],trupred[0][fr])
sx0_1,sy0_1,lax0_1,lay0_1,lgx0_1,lgy0_1,rax0_1,ray0_1,rgx0_1,rgy0_1=vizu.creat_skeleton(x_t,y_t)
sx1_1,sy1_1,lax1_1,lay1_1,lgx1_1,lgy1_1,rax1_1,ray1_1,rgx1_1,rgy1_1=vizu.creat_skeleton(x_0,y_0)
x_t,y_t,x_0,y_0=norm_from_spine2(target_aberman[0][fr],bfpred[0][fr])
sx0_2,sy0_2,lax0_2,lay0_2,lgx0_2,lgy0_2,rax0_2,ray0_2,rgx0_2,rgy0_2=vizu.creat_skeleton(x_t,y_t)
sx1_2,sy1_2,lax1_2,lay1_2,lgx1_2,lgy1_2,rax1_2,ray1_2,rgx1_2,rgy1_2=vizu.creat_skeleton(x_0,y_0)

ax1.plot(-sx0_1,-sy0_1, '--ko') 
ax1.plot(-lax0_1,-lay0_1, '--ko')
ax1.plot(-lgx0_1,-lgy0_1, '--ko')
ax1.plot(-rax0_1,-ray0_1, '--ko')
ax1.plot(-rgx0_1,-rgy0_1, '--ko')
ax1.plot(-sx1_1,-sy1_1, '-co') 
ax1.plot(-lax1_1,-lay1_1, '-ro')
ax1.plot(-lgx1_1,-lgy1_1, '-yo')
ax1.plot(-rax1_1,-ray1_1, '-bo')
ax1.plot(-rgx1_1,-rgy1_1, '-go')
ax1.set_aspect('equal')
ax1.set(xlim=(-0.1, 0.1), ylim=(-0.12, 0.1))
ax1.set_title('GCN reentrenado', fontsize=10)

ax2.plot(-sx0_2,-sy0_2, '--ko') 
ax2.plot(-lax0_2,-lay0_2, '--ko')
ax2.plot(-lgx0_2,-lgy0_2, '--ko')
ax2.plot(-rax0_2,-ray0_2, '--ko')
ax2.plot(-rgx0_2,-rgy0_2, '--ko')
ax2.plot(-sx1_2,-sy1_2, '-co') 
ax2.plot(-lax1_2,-lay1_2, '-ro')
ax2.plot(-lgx1_2,-lgy1_2, '-yo')
ax2.plot(-rax1_2,-ray1_2, '-bo')
ax2.plot(-rgx1_2,-rgy1_2, '-go')
ax2.set_aspect('equal')
ax2.set(xlim=(-0.1, 0.1), ylim=(-0.12, 0.1))
ax2.set_title('GCN reentrenado', fontsize=10)

plt.show()



images = []
filenames=list(range(0,64))
for filename in filenames:

    path=os.path.join('/home/chrisus/Desktop/Panoptics testing/imagesforgif/gcn/fr'+str(filename)+'.png')
    images.append(imageio.imread(path))
imageio.mimsave('/home/chrisus/Proyectofinal/TESIS/gcn.gif', images)

imagesar = []
for filename in filenames:

    path=os.path.join('/home/chrisus/Desktop/Panoptics testing/imagesforgif/4/Fir'+str(filename)+'.png')
    imagesar.append(imageio.imread(path))
imageio.mimsave('/home/chrisus/Proyectofinal/TESIS/fourmethods.gif', imagesar)
'''



imgs=[]
for i in range(0,64): 
    imgs.append('/home/chrisus/Desktop/Panoptics testing/imagesforgif/gcn/fr'+str(i)+'.png')
    print("scanned the image identified with",i)  

title_font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 22, encoding="unic")

cont=0

frames = []
for i in imgs:
    new_frame = Image.open(i)
    title_text = os.path.join("Cuadro " +str(cont))
    image_editable = ImageDraw.Draw(new_frame)
    image_editable.text((280,230), title_text, (10, 10, 10), font=title_font)
    cont=cont+1
    frames.append(new_frame)




frames[0].save('/home/chrisus/Proyectofinal/TESIS/gcnmethodsv2.gif', format='GIF',
    append_images=frames[1:],
    save_all=True,
    duration=200, loop=0)


