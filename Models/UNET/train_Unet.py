from seg_data_ import SegmentationDataset
from unet import UNet
import config
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import time
import os
import numpy as np
import wandb

#wandb.init(project="U-NET", entity="",resume=False) wandb account
'''
wandb.config = {
  "learning_rate": 0.001,
  "epochs": 50,
  "batch_size": 8
}
# load the image and mask filepaths in a sorted manner
'''
imput = config.IMAGE_DATASET_PATH 
omput= config.MASK_DATASET_PATH 



'''norm data for U-NET'''
def get_data(path):
	os.chdir(path)
	info = []
	w=[]
	print('reading encoded data')
	for file in os.listdir():
		input=np.load(file)
		s,t,d=np.shape(input)
		if d==4:
			rest=np.ones((128,8))*-9
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


def norm_matrix(datain,dataout):
    ns,_,_,_=np.shape(datain)
    scaled_in=np.copy(datain)
    scaled_out=np.copy(dataout)
    data=np.append(datain,dataout,axis=0)
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
        scaled_out[m][0] = (dataout[m][0]-opr*min_now)/(max_now-min_now)*10000

    return scaled_in,scaled_out,max_now,min_now

def transp(data):
    ns,m,hg,wd=np.shape(data)
    print('dimensions to transpose',ns,m,hg,wd)
    out=np.zeros((ns,wd,hg,m))
    for i in range(ns):
        out[i]=np.transpose(data[i])

    return out



imagePaths = get_data(imput)
maskPaths =  get_data(omput)  
normin,normout,max_now,min_now=norm_matrix(imagePaths,maskPaths)
inpdata=transp(normin)
outdata=transp(normout)
print(np.shape(inpdata),' vs ',np.shape(outdata))
inpdata=np.array(inpdata).astype(np.uint8)
outdata=np.array(outdata).astype(np.uint8)
# partition the data into training and testing splits using 85% of
# the data for training and the remaining 15% for testing
split = train_test_split(inpdata, outdata,
	test_size=config.TEST_SPLIT, random_state=5)
# unpack the data split
(trainImages, testImages) = split[:2]
(trainMasks, testMasks) = split[2:]
# write the testing image paths to disk so that we can use then
# when evaluating/testing our model
print("[INFO] saving testing image paths...")
f = open(config.TEST_PATHS, "w")
f.write("\n".join(str(testImages)))
f.close()

# define transformations
transforms = transforms.Compose([transforms.ToPILImage(),
 	transforms.Resize((config.INPUT_IMAGE_HEIGHT,
		config.INPUT_IMAGE_WIDTH)),
	transforms.ToTensor()])
# create the train and test datasets
trainDS = SegmentationDataset(imagePaths=trainImages, maskPaths=trainMasks,
	transforms=transforms)
testDS = SegmentationDataset(imagePaths=testImages, maskPaths=testMasks,
    transforms=transforms)
print(f"[INFO] found {len(trainDS)} examples in the training set...")
print(f"[INFO] found {len(testDS)} examples in the test set...")
# create the training and test data loaders
trainLoader = DataLoader(trainDS, shuffle=True,
	batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
	num_workers=os.cpu_count())
testLoader = DataLoader(testDS, shuffle=False,
	batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
	num_workers=os.cpu_count())

unet = UNet().to(config.DEVICE)
# initialize loss function and optimizer
lossFunc = BCEWithLogitsLoss()
opt = Adam(unet.parameters(), lr=config.INIT_LR)
# calculate steps per epoch for training and test set
trainSteps = len(trainDS) // config.BATCH_SIZE
testSteps = len(testDS) // config.BATCH_SIZE
# initialize a dictionary to store training history
H = {"train_loss": [], "test_loss": []}

print("[INFO] training the network...")
startTime = time.time()
for e in tqdm(range(config.NUM_EPOCHS)):
	# set the model in training mode
	unet.train()
	# initialize the total training and validation loss
	totalTrainLoss = 0
	totalTestLoss = 0
	# loop over the training set
	for (i, (x, y)) in enumerate(trainLoader):
		# send the input to the device
		(x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))
		# perform a forward pass and calculate the training loss
		pred = unet(x)
		loss = lossFunc(pred, y)
		# first, zero out any previously accumulated gradients, then
		# perform backpropagation, and then update model parameters
		opt.zero_grad()
		loss.backward()
		opt.step()
		# add the loss to the total training loss so far
		totalTrainLoss += loss
	# switch off autograd
	with torch.no_grad():
		# set the model in evaluation mode
		unet.eval()
		# loop over the validation set
		for (x, y) in testLoader:
			# send the input to the device
			(x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))
			# make the predictions and calculate the validation loss
			pred = unet(x)
			totalTestLoss += lossFunc(pred, y)
	# calculate the average training and validation loss
	avgTrainLoss = totalTrainLoss / trainSteps
	avgTestLoss = totalTestLoss / testSteps
	# update our training history
	H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
	H["test_loss"].append(avgTestLoss.cpu().detach().numpy())
	# print the model training and validation information
	#wandb.log({"Train loss": avgTrainLoss})
	#wandb.log({"Test loss": avgTestLoss})
	#wandb.save(os.path.join(wandb.run.dir, "checkpoint*"))
	print("[INFO] EPOCH: {}/{}".format(e + 1, config.NUM_EPOCHS))
	print("Train loss: {:.6f}, Test loss: {:.4f}".format(
		avgTrainLoss, avgTestLoss))
	
torch.save(unet.state_dict(), '/home/carguedas/saved_models/unetv1.pth')
# display the total time needed to perform the training
endTime = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(
	endTime - startTime))
os.chdir(os.path.join('/home/carguedas/saved_models/'))
desn=np.array([max_now,min_now],dtype=float)
np.savetxt('values_for_desnormalize',desn)

