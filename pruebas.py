from __future__ import absolute_import
from __future__ import print_function
import time
import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import math
import numpy as np
from torch.utils.data import DataLoader
from progressbar import Bar
from torch.autograd import Variable

class GraphConvolution(nn.Module):
    """
    adapted from : https://github.com/tkipf/gcn/blob/92600c39797c2bfb61a508e52b88fb554df30177/gcn/layers.py#L132
    """

    def __init__(self, in_features, out_features, bias=True, node_n=30):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.att = Parameter(torch.FloatTensor(node_n, node_n))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.att.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(self.att, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GC_Block(nn.Module):
    def __init__(self, in_features, p_dropout, bias=True, node_n=30):
        """
        Define a residual block of GCN
        """
        super(GC_Block, self).__init__()
        self.in_features = in_features
        self.out_features = in_features

        self.gc1 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn1 = nn.BatchNorm1d(node_n * in_features)

        self.gc2 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn2 = nn.BatchNorm1d(node_n * in_features)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

    def forward(self, x):
        y = self.gc1(x)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        y = self.gc2(y)
        b, n, f = y.shape
        y = self.bn2(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        return y + x

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, input_feature, hidden_feature, p_dropout, num_stage=1, node_n=48):
        """

        :param input_feature: num of input feature
        :param hidden_feature: num of hidden feature
        :param p_dropout: drop out prob.
        :param num_stage: number of residual blocks
        :param node_n: number of nodes in graph
        """
        super(GCN, self).__init__()
        self.num_stage = num_stage

        self.gc1 = GraphConvolution(input_feature, hidden_feature, node_n=node_n)
        self.bn1 = nn.BatchNorm1d(node_n * hidden_feature)

        self.gcbs = []
        for i in range(num_stage):
            self.gcbs.append(GC_Block(hidden_feature, p_dropout=p_dropout, node_n=node_n))

        self.gcbs = nn.ModuleList(self.gcbs)

        self.gc7 = GraphConvolution(hidden_feature, input_feature, node_n=node_n)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

    def forward(self, x):
        y = self.gc1(x)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        for i in range(self.num_stage):
            y = self.gcbs[i](y)

        y = self.gc7(y)
        y = y + x

        return y
        
def train(train_loader, model, optimizer, input_n=20, dct_n=30, lr_now=None, max_norm=True, is_cuda=False, dim_used=[]):
    t_l = AccumLoss()

    model.train()
    st = time.time()
    bar = Bar('>>>', fill='>', max=len(train_loader))
    for i, (inputs, targets, all_seq) in enumerate(train_loader):

        # skip the last batch if only have one sample for batch_norm layers
        batch_size = inputs.shape[0]
        if batch_size == 1:
            continue

        bt = time.time()
        if is_cuda:
            inputs = Variable(inputs.cuda()).float()
            # targets = Variable(targets.cuda(async=True)).float()
            all_seq = Variable(all_seq.cuda(async=True)).float()

        outputs = model(inputs)
        n = outputs.shape[0]
        outputs = outputs.view(n, -1)
        # targets = targets.view(n, -1)

        loss = sen_loss(outputs, all_seq, dim_used, dct_n)

        # calculate loss and backward
        optimizer.zero_grad()
        loss.backward()
        if max_norm:
            nn.utils.clip_grad_norm(model.parameters(), max_norm=1)
        optimizer.step()
        n, _, _ = all_seq.data.shape

        

        # update the training loss
        t_l.update(loss.cpu().data.numpy()[0] * n, n)


        bar.suffix = '{}/{}|batch time {:.4f}s|total time{:.2f}s'.format(i + 1, len(train_loader), time.time() - bt,
                                                                         time.time() - st)
        bar.next()
    bar.finish()
    return lr_now, t_l.avg

def get_dct_matrix(N):
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)
    return dct_m, idct_m


data = np.load('/home/chrisus/Proyectofinal/Motion/processed/S5/Directions 1.58860488.npy')

N=len(data[:])
print(N)
T=30#frames
x=[] #x,y coordinates

rowx=[]
rowy=[]
j=0 #joint

while j<15:
    for i in range(T):
        rowx=np.append(rowx,data[i][j][0])
        rowy=np.append(rowy,data[i][j][1])
    
    x.append(rowx)
    x.append(rowy)
    rowx=[]
    rowy=[]
    j+=1
print(np.shape(x))
d,i=get_dct_matrix(T)
input_featx=np.matmul(x,d)


start_epoch = 0
err_best = 10000
lr_now = 5.0e-4
is_cuda = torch.cuda.is_available()

input_n=26
output=4
sample_rate=2
model = GCN(input_feature=T, hidden_feature=256, p_dropout=0.5,
                        num_stage=12, node_n=48)
if is_cuda:
        model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=lr_now)

train_loader = DataLoader(
        dataset=input_featx,
        batch_size=16,
        shuffle=True,
        num_workers=10,
        pin_memory=True)
print(">>> data loaded !")
print(">>> train data {}".format(input_featx.__len__()))

x=[]
j=0
while j<15:
    for i in range(T):
        rowx=np.append(rowx,data[i+T][j][0])
        rowy=np.append(rowy,data[i+T][j][1])
    
    x.append(rowx)
    x.append(rowy)
    rowx=[]
    rowy=[]
    j+=1
print(np.shape(x))
d,i=get_dct_matrix(T)
val_dataset=np.matmul(x,d)

val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=10,
        pin_memory=True)

print(">>> validation data {}".format(val_dataset.__len__()))
lr_decay=2
lr_gamma=0.96

for epoch in range(start_epoch, 50):

        if (epoch + 1) % lr_decay == 0:
            lr_now = lr_decay(optimizer, lr_now, lr_gamma)
        print('==========================')
        print('>>> epoch: {} | lr: {:.5f}'.format(epoch + 1, lr_now))
        ret_log = np.array([epoch + 1])
        head = np.array(['epoch'])
        # per epoch
        lr_now, t_l= train(train_loader, model, optimizer, input_n=input_n,
                                       lr_now=lr_now, max_norm=True, is_cuda=is_cuda,
                                       dim_used=2, dct_n=30)
        ret_log = np.append(ret_log, [lr_now, t_l])
        head = np.append(head, ['lr', 't_l'])

        v_e, v_3d = val(val_loader, model, input_n=input_n, is_cuda=is_cuda, dim_used=2,
                        dct_n=30)

        ret_log = np.append(ret_log, [v_e, v_3d])
        head = np.append(head, ['v_e', 'v_3d'])
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



class AccumLoss(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

def sen_loss(outputs, all_seq, dim_used, dct_n):
    """

    :param outputs: N * (seq_len*dim_used_len)
    :param all_seq: N * seq_len * dim_full_len
    :param input_n:
    :param dim_used:
    :return:
    """
    n, seq_len, dim_full_len = all_seq.data.shape
    dim_used_len = len(dim_used)
    dim_used = np.array(dim_used)

    _, idct_m = get_dct_matrix(seq_len)
    idct_m = Variable(torch.from_numpy(idct_m)).float().cuda()
    outputs_t = outputs.view(-1, dct_n).transpose(0, 1)
    pred_expmap = torch.matmul(idct_m[:, :dct_n], outputs_t).transpose(0, 1).contiguous().view(-1, dim_used_len,
                                                                                               seq_len).transpose(1, 2)
    targ_expmap = all_seq.clone()[:, :, dim_used]

    loss = torch.mean(torch.sum(torch.abs(pred_expmap - targ_expmap), dim=2).view(-1))
    return loss