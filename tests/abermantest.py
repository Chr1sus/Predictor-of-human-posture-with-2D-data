from itertools import count
import numpy as np
import os,sys
import torch
import Models.UNET.config as config
from Models.UNET.seg_data_ import SegmentationDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import ops.vizu
import imageio
from PIL import Image,ImageFont, ImageDraw 
import glob  #use it if you want to read all of the certain file type in the directory

input_data_target=np.load('input_aberman_test_for_att_unet_.npy')
print(input_data_target)
print('shape',np.shape(input_data_target))
print('max value',np.max(input_data_target),'min value',np.min(input_data_target))
print('max value abs',np.max(np.absolute(input_data_target)),'min value',np.min(np.absolute(input_data_target)))





imgs=[]
for i in range(0,64): 
    imgs.append('/dir_of_images/'+str(i)+'.png')
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




frames[0].save('/add_text_to_images/', format='GIF',
    append_images=frames[1:],
    save_all=True,
    duration=200, loop=0)


