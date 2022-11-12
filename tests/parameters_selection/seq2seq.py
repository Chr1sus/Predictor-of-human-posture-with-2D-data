import torch
import torch.nn as nn
import torch.optim as opt
import matplotlib.pyplot as plt
import matplotlib.ticker as tck

import numpy as np
import random
import math
import time
import os

from Models.ATTN.ATTN import Encoder,EncoderLayer,Decoder,DecoderLayer, MultiHeadAttentionLayer, Seq2Seq

import wandb

class train_seq():
    def __init__(self) -> None:
         pass

        
    def hyper(self,lrinit,dropout):
        self.LEARNING_RATE=lrinit
        self.dropout=dropout
        
        

    def load_data(self): 
        def prep_data(dir):
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
        p32='path_coded_data_incomplete'
        p64='path_coded_data_complete'

        self.in32=prep_data(p32)
        self.in64=prep_data(p64)


    def init_model(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #wandb.init(project="attention", entity="")
        INPUT_DIM=8
        OUTPUT_DIM=8
        HID_DIM = 8
        ENC_LAYERS = 3
        DEC_LAYERS = 3
        ENC_HEADS = 4
        DEC_HEADS = 4
        ENC_PF_DIM = 256
        DEC_PF_DIM = 256
        ENC_DROPOUT = self.dropout
        DEC_DROPOUT = self.dropout

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

        self.model = Seq2Seq(enc, dec, self.in32, self.in64, device).to(device)

        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f'The model has {count_parameters(self.model):,} trainable parameters')

        def initialize_weights(m):
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                nn.init.xavier_uniform_(m.weight.data)

        self.model.apply(initialize_weights)

    def proccess(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr = self.LEARNING_RATE)

        #criterion = nn.CrossEntropyLoss()
        criterion = nn.MSELoss()
        def train(model, iterator, optimizer, criterion, clip,in32,in64):
            
            model.train()
            epoch_loss = 0
            
            for i in range(16,iterator,16):
                
                src = in32[i-16:i]
                trg = in64[i-16:i]
                src = torch.FloatTensor(src)
                trg = torch.FloatTensor(trg)
                
                optimizer.zero_grad()
                
                output, _ = model(src.cuda(), trg.cuda())
                        
                #output = [batch size, trg len - 1, output dim]
                #trg = [batch size, trg len]
                    
                output_dim = output.shape[-1]
                    
                output = output.contiguous().view(-1, output_dim)
                trg = trg.contiguous().view(-1, output_dim)
                        
                #output = [batch size * trg len - 1, output dim]
                #trg = [batch size * trg len - 1]
                #trg=torch.tensor(trg, dtype=torch.long, device=device)
                #trg=trg.to(device=device, dtype=torch.int64)
                loss = criterion(output, trg.cuda())
                wandb.log({"train loss": loss})
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                
                optimizer.step()
                
                epoch_loss += loss.item()
            wandb.log({"epoch training loss": epoch_loss})
                
            return epoch_loss/iterator

        def evaluate(model, iterator, criterion,in32,in64):
            
            model.eval()
            
            epoch_loss = 0
            
            with torch.no_grad():
            
                for i in range(15376,iterator,16):

                    src = in32[i-16:i]
                    trg = in64[i-16:i]
                    src = torch.FloatTensor(src)
                    trg = torch.FloatTensor(trg)
                    output, _ = model(src.cuda(), trg.cuda())
                    
                    #output = [batch size, trg len - 1, output dim]
                    #trg = [batch size, trg len]
                    
                    output_dim = output.shape[-1]
                    
                    output = output.contiguous().view(-1, output_dim)
                    trg = trg.contiguous().view(-1, output_dim)
                    
                    #output = [batch size * trg len - 1, output dim]
                    #trg = [batch size * trg len - 1]
                    
                    loss = criterion(output, trg.cuda())

                    wandb.log({"val loss": loss})

                    epoch_loss += loss.item()
            
            wandb.log({"epoch validation loss": epoch_loss})
            return epoch_loss / iterator

        def epoch_time(start_time, end_time):
            elapsed_time = end_time - start_time
            elapsed_mins = int(elapsed_time / 60)
            elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
            return elapsed_mins, elapsed_secs

        N_EPOCHS = 10
        CLIP = 1
        wandb.config = {
        "learning_rate": 0.001,
        "epochs": 10,
        "batch_size": 16
        }


        best_valid_loss = float('inf')

        for epoch in range(N_EPOCHS):
            
            start_time = time.time()
            train_iterator=15360
            valid_iterator=16512
            train_loss = train(self.model, train_iterator, optimizer, criterion, CLIP,self.in32,self.in64)
            valid_loss = evaluate(self.model, valid_iterator, criterion,self.in32,self.in64)
            
            end_time = time.time()
            
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                #torch.save(self.model.state_dict(), 'tut6-model.pt')
            
            print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
            wandb.log({"train loss per epoch": train_loss})
            wandb.log({"validation loss per epoch": valid_loss})
            
        #model.load_state_dict(torch.load('tut6-model.pt'))

        return train_loss,valid_loss


def main():
    seq=train_seq()
    init_lr =[0.0001,0.001, 0.01]
    drop =[0.0001,0.001, 0.01,0.1]

    seq.load_data()
    
    tr=[]
    ts=[]
    ls=[]
    bs=[]
    for l in range(len(init_lr)):
        for b in range(len(drop)):
            seq.hyper(lrinit=init_lr[l],dropout=drop[b])
            seq.init_model()
            train,test=seq.proccess()
            tr.append(train)
            ts.append(test)
            ls.append(init_lr[l])
            bs.append(drop[b])
    print('train data')
    print(tr)
    print("#################################")
    print("#################################")
    print("#################################")
    print('test data')
    print(ts)
    print("#################################")
    print("#################################")
    print("#################################")
    print('learning rate')
    print(ls)
    print("#################################")
    print("#################################")
    print("#################################")
    print('drop out')
    print(bs)
    print("#################################")
    print("#################################")
    print("#################################")

			
if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main()
