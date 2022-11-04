from __future__ import absolute_import
from __future__ import print_function
import time
import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import math
import numpy as np
from torch.utils.data import DataLoader
from progress.bar import Bar
from torch.autograd import Variable
import wandb
import os
from ops.loaddata import load_data_qdrcn
from ops.loaddata import norm
from GCN_DCT import get_dct_matrix,AccumLoss,mpjpe_error_p3d,GCN,lr_decay

#wandb.init(project="", entity=,resume=False) for wandb account
'''
wandb.config = {
  "learning_rate": 0.001,
  "epochs": 50,
  "batch_size": 16
}
'''
# Log metrics inside your training loop to visualize model performance
#wandb.log({"loss": loss})

# Optional
#wandb.watch(model)
torch.cuda.empty_cache()
torch.cuda.memory_summary(device=None, abbreviated=False)



def train(train_loader, model, optimizer, input_n=32, dct_n=64, lr_now=None, max_norm=True, is_cuda=True, dim_used=[]):
    t_l = AccumLoss()

    model.train()
    st = time.time()
    bar = Bar('>>>', fill='>', maxval=len(train_loader))

    for i, inputs in enumerate(train_loader):
        #print(i,'i')
        #print(np.shape(inputs),'inputs')
        all_seq=inputs
        # skip the last batch if only have one sample for batch_norm layers
        batch_size = inputs.shape[0]
        if batch_size == 1:
            continue

        bt = time.time()
        input=inputs.float()
        outputs = model(input.cuda())
        n = outputs.shape[0]
        outputs = outputs.view(n, -1)
        # targets = targets.view(n, -1)


        loss = mpjpe_error(outputs.cuda(), all_seq.cuda())

        # calculate loss and backward
        optimizer.zero_grad()
        loss.backward()
        if max_norm:
            nn.utils.clip_grad_norm(model.parameters(), max_norm=1)
        optimizer.step()
        n = all_seq.data.shape

        

        # update the training loss
        t_l.update(loss.cpu().data.numpy() * n, len(n))


        bar.suffix = '{}/{}|batch time {:.4f}s|total time{:.2f}s'.format(i + 1, len(train_loader), time.time() - bt,
                                                                         time.time() - st)
        bar.next()
    bar.finish()
    return lr_now, t_l.avg

def mpjpe_error(batch_pred,batch_gt): 
    batch_pred=batch_pred.contiguous().view(-1,30)
    batch_gt=batch_gt.contiguous().view(-1,30)

    return torch.mean(torch.norm(batch_gt-batch_pred,2,1))


def test(train_loader, model, input_n=32, output_n=32, dct_n=64, is_cuda=False, dim_used=[]):
    N = 0
    # t_l = 0
    if output_n >= 25:
        eval_frame = [1, 3, 7, 9, 13, 24]
    

    t_3d = np.zeros(len(eval_frame))

    model.eval()
    st = time.time()
    bar = Bar('>>>', fill='>', max=len(train_loader))
    for i, inputs in enumerate(train_loader):
        bt = time.time()
        all_seq=inputs

        outputs = model(inputs.cuda())
        n = outputs.shape[0]
        # outputs = outputs.view(n, -1)
        # targets = targets.view(n, -1)

        # loss = loss_funcs.sen_loss(outputs, all_seq, dim_used)

        n, dim_full_len = all_seq.data.shape
        dim_used_len = dim_used

        # inverse dct transformation
        _, idct_m =get_dct_matrix(dim_used)
        idct_m = Variable(torch.from_numpy(idct_m)).float().cuda()
        outputs_t = outputs.view(-1, dct_n).transpose(0, 1)
        np_arr = idct_m.cpu().detach().numpy()
        np_ass = outputs_t.cpu().detach().numpy()
        resul=np.matmul(np_arr,np_ass)
        pred_p3d=torch.from_numpy(resul)
    
    #pred_expmap = resul_t.transpose(0, 1).contiguous().view(-1, dim_used_len )
        targ_p3d = all_seq.clone().transpose(0,1)

        # update loss and testing errors
        for k in np.arange(0, len(eval_frame)):
            j = eval_frame[k]
            t_3d[k] += torch.mean(torch.norm(
                targ_p3d[:, j, :, :].contiguous().view(-1, 3) - pred_p3d[:, j, :, :].contiguous().view(-1, 3), 2,
                1)).cpu().data.numpy()[0] * n
        # t_l += loss.cpu().data.numpy()[0] * n
        N += n

        bar.suffix = '{}/{}|batch time {:.4f}s|total time{:.2f}s'.format(i + 1, len(train_loader), time.time() - bt,
                                                                         time.time() - st)
        bar.next()
    bar.finish()
    return  t_3d / N



def val(train_loader, model, input_n=32, dct_n=64, is_cuda=False, dim_used=[]):
    # t_l = utils.AccumLoss()
    t_3d = AccumLoss()

    model.eval()
    st = time.time()
    bar = Bar('>>>', fill='>', max=len(train_loader))
    for i, inputs in enumerate(train_loader):
        bt = time.time()
        all_seq=inputs
        
        input=inputs.float()
        outputs = model(input.cuda())
        n = outputs.shape[0]
        outputs = outputs.view(n, -1)
        # targets = targets.view(n, -1)

        # 
        n= all_seq.data.shape
    
        loss = mpjpe_error(outputs.cuda(), all_seq.cuda())


        t_3d.update(loss.cpu().data.numpy()* n, len(n))

        bar.suffix = '{}/{}|batch time {:.4f}s|total time{:.2f}s'.format(i + 1, len(train_loader), time.time() - bt,
                                                                         time.time() - st)
        bar.next()
    bar.finish()
    return t_3d.avg



T=64
typ='train'
x=load_data_qdrcn(typ,T)
x=norm(x)
ns,fr,jo=np.shape(x)
d,i=get_dct_matrix(T)
#x=np.transpose(x)
train_data=[]
for i in range(ns):
    input_train=np.matmul(np.transpose(x[i]),d)
    train_data.append(input_train)


start_epoch = 0
err_best = 10000
lr_now = 5.0e-4
is_cuda = torch.cuda.is_available()

input_n=32
output=32
sample_rate=2
model = GCN(input_feature=T, hidden_feature=30, p_dropout=0.5,
                        num_stage=12, node_n=30)
if is_cuda:
        model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=lr_now)
print('2',np.shape(train_data))
train_loader = DataLoader(
        dataset=train_data,
        batch_size=30,
        shuffle=True,
        num_workers=10,
        pin_memory=True)

torch.save(model.state_dict(),'save_dir/')

print(">>> data loaded !")
print(">>> train data {}".format(input_train.__len__()))

typ='val'
x=load_data_qdrcn(typ,T)
x=norm(x)
ns,fr,jo=np.shape(x)
val_dataset=[]
for i in range(ns):
    input_train=np.matmul(np.transpose(x[i]),d)
    val_dataset.append(input_train)


val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=30,
        shuffle=False,
        num_workers=10,
        pin_memory=True)

print(">>> validation data {}".format(val_dataset.__len__()))
lr_gamma=0.96
lr_deca=2
acts='test'
x=load_data_qdrcn(typ,T)
test_data = dict()
ns,fr,jn=np.shape(x)
for act in range(ns):
    test_data[act] = DataLoader(
            dataset=x[act],
            batch_size=30,
            shuffle=False,
            num_workers=10,
            pin_memory=True)

 
if wandb.run.resumed:
    checkpoint = torch.load(wandb.restore(os.path.join(wandb.run.dir, "checkpoint*")))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']


for epoch in range(start_epoch, 50):

        if (epoch + 1) % lr_deca == 0:
            lr_now = lr_decay(optimizer, lr_now, lr_gamma)
        print('==========================')
        print('>>> epoch: {} | lr: {:.5f}'.format(epoch + 1, lr_now))
        ret_log = np.array([epoch + 1])
        head = np.array(['epoch'])
        # per epoch
        lr_now, t_l= train(train_loader, model, optimizer, input_n=input_n,
                                       lr_now=lr_now, max_norm=True, is_cuda=is_cuda,
                                       dim_used=2, dct_n=30)

        #wandb.log({"learnin rate": lr_now})
        #wandb.log({"loss avarage": t_l})
        ret_log = np.append(ret_log, [lr_now, t_l])
        
        head = np.append(head, ['lr', 't_l'])
        v_3d = val(val_loader, model, input_n=input_n, is_cuda=is_cuda, dim_used=2,
                        dct_n=64)
        ret_log = np.append(ret_log, [v_3d])
        head = np.append(head, ['v_3d'])  
        #wandb.log({"loss info": v_3d})
        #wandb.save(os.path.join(wandb.run.dir, "checkpoint*"))
torch.save(model.state_dict(), 'save_dir/')
#artifact = wandb.Artifact('model', type='model')
#artifact.add_file('')
#run=wandb.init(project="", entity="")
#run.log_artifact(artifact)
#run.join()
'''
        test_3d_temp = np.array([])
        test_3d_head = np.array([])
        
        for act in range(len(x[:])):
            test_3d = test(test_data[act], model, input_n=input_n, output_n=32, is_cuda=is_cuda,
                                   dim_used=2, dct_n=64)
            test_3d_temp = np.append(test_3d_temp, test_3d)
            test_3d_head = np.append(test_3d_head,[act + '3d80', act + '3d160', act + '3d320', act + '3d400'])
            head = np.append(head, [act + '80', act + '160', act + '320', act + '400'])
            
            head = np.append(head, [act + '560', act + '1000'])
            test_3d_head = np.append(test_3d_head,
                                         [act + '3d560', act + '3d1000'])
            ret_log = np.append(ret_log, test_3d_temp)
            head = np.append(head, test_3d_head)
'''

        
        #best_model = wandb.restore('model-best.h5', run_path="vanpelt/my-project/a1b2c3d")

'''
        test_3d_temp = np.array([])
        test_3d_head = np.array([])
        for act in acts:
            test_e, test_3d = test(test_data[act], model, input_n=input_n, output_n=output_n, is_cuda=is_cuda,
                                   dim_used=train_dataset.dim_used, dct_n=dct_n)
            ret_log = np.append(ret_log, test_e)
            test_3d_temp = np.append(test_3d_temp, test_3d)
            test_3d_head = np.append(test_3d_head,
                                     [act + '3d80', act + '3d160', act + '3d320', act + '3d400'])
            head = np.append(head, [act + '80', act + '160', act + '320', act + '400'])
            if output_n > 10:
                head = np.append(head, [act + '560', act + '1000'])
                test_3d_head = np.append(test_3d_head,
                                         [act + '3d560', act + '3d1000'])
        ret_log = np.append(ret_log, test_3d_temp)
        head = np.append(head, test_3d_head)
'''
        # update log file and save checkpoint
        
'''
        df = pd.DataFrame(np.expand_dims(ret_log, axis=0))
        if epoch == start_epoch:
            df.to_csv('checkpoint' + '/' + script_name + '.csv', header=head, index=False)
        else:
            with open('checkpoint'  + '/' + script_name + '.csv', 'a') as f:
                df.to_csv(f, header=False, index=False)
        if not np.isnan(v_e):
            is_best = v_e < err_best
            err_best = min(v_e, err_best)
        else:
            is_best = False
        file_name = ['ckpt_' + script_name + '_best.pth.tar', 'ckpt_' + script_name + '_last.pth.tar']
        utils.save_ckpt({'epoch': epoch + 1,
                         'lr': lr_now,
                         'state_dict': model.state_dict(),
                         'optimizer': optimizer.state_dict()},
                        ckpt_path='checkpoint' ,
                        is_best=is_best,
                        file_name=file_name)
'''



