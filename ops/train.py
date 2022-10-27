from glob import glob
from importlib.resources import path
from macpath import dirname
from multiprocessing.resource_sharer import stop
from scipy.ndimage.filters import gaussian_filter1d
import tqdm
import torch
import wandb
import numpy as np
import random
import os, sys
import pandas as pd 

from predictor_utils.arg_parser import args
from predictor_utils.visualize_pose_sequence \
    import draw_poses_cv2, draw_multiple_poses_cv2
from predictor_utils.general_utils import \
    translate_poses_for_motion_retargeter, \
    translate_poses_from_motion_retargeter
from loaddata import load_data_qdrcn as ld
from Motion_Retargeting.common import config
from Motion_Retargeting.model import get_autoencoder
from Motion_Retargeting.dataset import get_meanpose
from Motion_Retargeting.functional.utils import pad_to_height
from Motion_Retargeting.functional.motion import \
    preprocess_motion2d as normalize_and_move_poses_to_torch, \
    postprocess_motion2d as denormalize_and_move_poses_to_numpy

SAMPLE_FILES = [
    'panoptic_171026_pose3body_0_theta1.69_phi1.11_radius275.npy',
    'panoptic_171026_pose3body_0_theta0.83_phi5.40_radius275.npy',
    'panoptic_171204_pose5body_3_theta0.95_phi0.91_radius275.npy',
    'panoptic_171204_pose5body_3_theta0.90_phi2.13_radius275.npy',
    'panoptic_171204_pose1body_1_theta1.91_phi5.05_radius275.npy',
]

class TestMotionRetargeter:

    def __init__(s, args):
        s.config = config
        s.config.initialize(args)

        s.net = get_autoencoder(s.config)
        s.net.load_state_dict(torch.load(args.model_path))
        s.net.to(s.config.device)
        s.net.eval()

        s.h_motion, s.w_motion, s.scale_motion = pad_to_height(s.config.img_size[0], 480, 640)
        s.h_skeleton, s.w_skeleton, s.scale_skeleton = pad_to_height(s.config.img_size[0], 480, 640)
        s.h_view, s.w_view, s.scale_view = pad_to_height(s.config.img_size[0], 480, 640)

        s.mean_pose, s.std_pose = get_meanpose(s.config)

    def load_pose_file(s, file_names, lengths = [64, 64, 64], starts = [200, 0, 128]):

        poses_motion = np.load(file_names[0])   [starts[0]: starts[0] + lengths[0]]
        poses_skeleton = np.load(file_names[1]) [starts[1]: starts[1] + lengths[1]]
        poses_view = np.load(file_names[2])     [starts[2]: starts[2] + lengths[2]]

        return poses_motion, poses_skeleton, poses_view

    def retarget_motion(s, poses_motion, poses_skeleton, poses_view):
        
        input_motion = translate_poses_for_motion_retargeter(poses_motion, s.scale_motion)
        input_skeleton = translate_poses_for_motion_retargeter(poses_skeleton, s.scale_skeleton)
        input_view = translate_poses_for_motion_retargeter(poses_view, s.scale_view)

        n_input_motion = normalize_and_move_poses_to_torch(input_motion, s.mean_pose, s.std_pose)
        n_input_skeleton = normalize_and_move_poses_to_torch(input_skeleton, s.mean_pose, s.std_pose)
        n_input_view = normalize_and_move_poses_to_torch(input_view, s.mean_pose, s.std_pose)

        n_input_motion = n_input_motion.to(s.config.device)
        n_input_skeleton = n_input_skeleton.to(s.config.device)
        n_input_view = n_input_view.to(s.config.device)

        #out = net.transfer_three(n_input_motion, n_input_skeleton, n_input_view)

        m1 = s.net.mot_encoder(n_input_motion)
        b2 = s.net.body_encoder(n_input_skeleton[:, :-2, :]).repeat(1, 1, m1.shape[-1])
        v3 = s.net.view_encoder(n_input_view[:, :-2, :]).repeat(1, 1, m1.shape[-1])
        
        out = s.net.decoder(torch.cat([m1, b2, v3], dim=1))

        out = denormalize_and_move_poses_to_numpy(out, s.mean_pose, s.std_pose, s.w_skeleton//2, s.h_skeleton//2)

        out = translate_poses_from_motion_retargeter(out)

        return out, m1, b2, v3

    def encode_motion(s, poses_motion):
        input_motion = translate_poses_for_motion_retargeter(poses_motion, s.scale_motion)
        n_input_motion = normalize_and_move_poses_to_torch(input_motion, s.mean_pose, s.std_pose)
        n_input_motion = n_input_motion.to(s.config.device)
        m1 = s.net.mot_encoder(n_input_motion)
        
        return m1

    def retarget_motion_online(s, poses_motion, poses_skeleton, poses_view, pipeline_length=64):
        out = np.empty((0, 15, 2))
        for i in tqdm.tqdm(range(pipeline_length,len(poses_motion))):
            pipeline = poses_motion[i-pipeline_length:i]
            pipeline = np.flip(pipeline, axis=0)
            out2, *_ = s.retarget_motion(pipeline, poses_skeleton, poses_view)
            out = np.append(out, out2[np.newaxis, 0], axis=0)
            temp = gaussian_filter1d(out, sigma=2, axis=0)
            out[-1] = temp[-1]
        return out

    def run_flipped_pose_mock_online(s, pose_files):

        skeleton_file = pose_files[1]
        view_file = pose_files[2]

        save_file = 'flipped_pose_experiment_mock_online_without_filtering.csv'
        with open(save_file, 'w') as f:
            f.write(f'Skeleton file: {skeleton_file}\n')
            f.write(f'View file: {view_file}\n')

        for pose_file in pose_files[:10]:
            motion_file = pose_file
            print(f'Motion file: {motion_file}')

            poses_motion, poses_skeleton, poses_view = s.load_pose_file(
                file_names = [motion_file, skeleton_file, view_file],
                lengths = [512, 64, 64],
                starts = [0, 0, 0]
            )
            out, *_ = s.retarget_motion(
                poses_motion,
                poses_skeleton,
                poses_view
            )
            out = out[64:]

            out2 = s.retarget_motion_online(
                poses_motion,
                poses_skeleton,
                poses_view
            )

            # out = out - out[:, np.newaxis, 2,:] + [320, 240]
            # out2 = out2 - out2[:, np.newaxis, 2, :] + [320, 240]
            # diff = out - out2
            # draw_multiple_poses_cv2(
            #     [out, out2, diff + [320, 240]],
            #     ['normal', 'flipped', 'diff'],
            #     scale=1,
            #     translation=[0,0],
            #     wait_time=30,
            #     pose_fmt='cmu'
            # )
            
            with open(save_file, 'a') as f:
                f.write(f'Experiment for: {motion_file}\n')
                f.write('average joint, average pose\n')

                out = out - out[:, np.newaxis, 2, :]
                out2 = out2 - out2[:, np.newaxis, 2, :]
                error = out - out2
                avg1 = np.average(np.average(error, axis=1), axis=0)
                avg2 = np.average(error, axis=0)

                line = f'{avg1}, ' + ', '.join([str(elem) for elem in avg2])
                line = line + '\n'
                f.write(line)

    def generate_encoded_files(s, pose_files, output_dir):
        print('generate encoded files')
        dir_name32='/home/chrisus/Proyectofinal/Motion/disptec-2020/tools/motion_predictor/output/final/testing/'
        dir_name64='/home/chrisus/Proyectofinal/Motion/disptec-2020/tools/motion_predictor/output/final/testing/'
        #dir_name32='/home/chrisus/Proyectofinal/Motion/disptec-2020/tools/motion_predictor/vw/'
        #dir_name64='/home/chrisus/Proyectofinal/Motion/disptec-2020/tools/motion_predictor/sk/'
        suffix='.npy'
        #s.h_motion, s.w_motion, s.scale_motion = pad_to_height(512, 480, 640)
        #s.mean_pose, s.std_pose = get_meanpose(s.config)
        #s.net = get_autoencoder(s.config)
        #s.net.load_state_dict(torch.load(args.model_path))
        #s.net.to(s.config.device)
        #s.net.eval()
        sample,fps,joint,xy=np.shape(pose_files)
        print('samples',sample)
        rf=list(range(0, sample))
        rfsk=list(map(str,rf))
        print(rfsk)
        #result=np.zeros(( sample,fps,joint,xy))
        result=np.zeros((64,15,2))
        print('final result shape',np.shape(result))
        for i in range(sample):
            b2=torch.load(dir_name64+'sf0.pt')
            v3=torch.load(dir_name32+'vf0.pt')
            #b2=torch.load(dir_name64+'sa'+rfsk[0]+'.pt')
            #v3=torch.load(dir_name32+'va'+rfsk[0]+'.pt')
            '''
            input_skeleton = translate_poses_for_motion_retargeter(pose_files[i], s.scale_skeleton)
            input_view = translate_poses_for_motion_retargeter(pose_files[i], s.scale_view)
            n_input_skeleton = normalize_and_move_poses_to_torch(input_skeleton, s.mean_pose, s.std_pose)
            n_input_view = normalize_and_move_poses_to_torch(input_view, s.mean_pose, s.std_pose)

        
            n_input_skeleton = n_input_skeleton.to(s.config.device)
            n_input_view = n_input_view.to(s.config.device)
            
            m1 = s.encode_motion(s,pose_files[i])
            b2 = s.net.body_encoder(n_input_skeleton[:, :-2, :]).repeat(1, 1, m1.shape[-1])
            v3 = s.net.view_encoder(n_input_view[:, :-2, :]).repeat(1, 1, m1.shape[-1])
            '''
            #m1 = s.encode_motion(s,pose_files[i])
            m1=torch.Tensor(pose_files[i])
            out = s.net.decoder(torch.cat([m1.cuda(), b2.cuda(), v3.cuda()], dim=1))
            
            out = denormalize_and_move_poses_to_numpy(out, s.mean_pose, s.std_pose, s.w_skeleton//2, s.h_skeleton//2)

            out = translate_poses_from_motion_retargeter(out)
            #result[i]=out
            result=out
        print(np.shape(result))
        #np.save('/home/chrisus/Proyectofinal/Motion/disptec-2020/tools/motion_predictor/output/target_test.npy',result)
        np.save(output_dir,result)
        '''
        for (dirpath, dirnames,pose_file) in os.walk(pose_files):
            print(dirpath,dirnames,pose_file)
            for p in range(len(pose_file)):
                poses = np.load(pose_files+pose_file[p])
                ref=str(dirpath[45:48])
                print(np.shape(poses))
                for i in range(128, len(poses), 128):
                    poses64 = poses[i-128:i]
                    pos=[]
                    for d in range(64):
                        pos.append(poses64[d*2])
                    poses32 = pos[:32]
                    basename=pose_file[p]  
                    m64 = s.encode_motion(s,pos)
                    m32 = s.encode_motion(s,poses32)
                    t64=m64.detach().cpu().numpy()
                    t32=m32.detach().cpu().numpy()
                    begin=str(i-128)
                    end=str(i)
                    s32=os.path.join(dir_name32,ref+ basename+"_"+begin+'to'+end + suffix)
                    s64=os.path.join(dir_name64,ref+ basename +"_"+begin+'to'+end + suffix)
                    with open(s32, 'wb') as f:
                        np.save(f, t32)
                    with open(s64, 'wb') as f:
                        np.save(f, t64)

                    #torch.save(m64, output_dir + output_name_m64)
                    #torch.save(m32, output_dir + output_name_m32)
        
        
        samples,_,_,_=np.shape(pose_files)
        print('samples: ',samples)
        for p in range(samples):
            poses32 = pose_files[p]  
            m32 = s.encode_motion(s,poses32)
            t32=m32.detach().cpu().numpy()
            begin=str(p)
            s32=os.path.join(dir_name32,'data_'+begin+ suffix)
            with open(s32, 'wb') as f:
                np.save(f, t32)
        '''

        


class PosePredictor(torch.nn.Module):

    def __init__(self):
        super(PosePredictor, self).__init__()

        activation = torch.nn.LeakyReLU(0.2)
        conv = torch.nn.Conv1d
        upsample = torch.nn.Upsample(
            scale_factor=2, mode='nearest'
        )

        model = []  
        model.append(conv(128, 64, 4))
        model.append(activation)
        model.append(upsample)
        model.append(conv(64, 128, 3))
        model.append(activation)

        self.model = torch.nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        return x

class PosePredictorTrainer:

    def __init__(s):
        s.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        s.criterion = torch.nn.MSELoss()
        s.learning_rate = 0.001
        s.epochs = 1000
        s.batch_size = 1
        s.optimizer_function = torch.optim.Adam
    
    def get_optimizer(s, model):
        s.optimizer = s.optimizer_function(model.parameters(), lr = s.learning_rate)
        return s.optimizer
    
    def initialize_wandb(s, model):
        wandb.init(
            name='test_run',
            project="disptec2021-motion_predictor",
            entity="franguindon"
        )
        wandb.config = {
            "learning_rate": s.learning_rate,
            "epochs": s.epochs,
            "batch_size": s.batch_size
        }
        wandb.watch(model)

    def get_epoch_iterator(s):
        epoch_iterator = range(s.epochs)
        progress_bar = tqdm.tqdm(epoch_iterator)
        return progress_bar

class EncodedPosesDataset(torch.utils.data.Dataset):
    
    def __init__(s):
        s.encoder_dir = '/media/francis/ADATA HV300/encoded_motion_data/'
        s.encoder_files = glob(s.encoder_dir + '/*m32from*')

    def __len__(s):
        return len(s.encoder_files)

    def __getitem__(s, index):
        data_file = s.encoder_files[index]
        label_file = data_file.replace('m32from', 'm64from')
        data = torch.load(data_file)
        label = torch.load(label_file)
        return data, label

def main():
    #encoder_dir = '/media/francis/ADATA HV300/encoded_motion_data/'
    #encoder_files = glob(encoder_dir + '/*m32from*')
    #random.seed(20)
    #random.shuffle(encoder_files)
    
    #num_files = len(encoder_files)
    #percent_train_files = 70
    #num_train_files = num_files*percent_train_files//100
    #train_files = encoder_files[:num_train_files]
    #valid_files = encoder_files[num_train_files:]
    
    #trainer = PosePredictorTrainer()
    #device = trainer.device
    #pose_predictor = PosePredictor()
    #print(pose_predictor)
    #print(f'Moving pose predictor to {device} ...')
    #pose_predictor.to(device)
    
    #criterion = trainer.criterion
    #optimizer = trainer.get_optimizer(pose_predictor)
    
    #trainer.initialize_wandb(pose_predictor)
    
    #epoch_it = trainer.get_epoch_iterator()
    '''
    for epoch in epoch_it:
        pose_predictor.train()
        train_file = train_files[epoch]
        train_label_file = train_file.replace('m32from', 'm64from')
        train_data = torch.load(train_file)
        train_label = torch.load(train_label_file)
        
        optimizer.zero_grad()
        output_data = pose_predictor(train_data)
        train_loss = criterion(output_data, train_label)
        train_loss.backward()
        optimizer.step()

        pose_predictor.eval()
        valid_file = valid_files[0]
        valid_label_file = valid_file.replace('m32from', 'm64from')
        valid_data = torch.load(valid_file)
        valid_label = torch.load(valid_label_file)

        output_data = pose_predictor(valid_data)
        valid_loss = criterion(output_data, valid_label)

        wandb.log({
            "Epoch": epoch,
            "Train Loss": train_loss,
            "Valid Loss": valid_loss
        })
        '''
    T=TestMotionRetargeter.__init__(TestMotionRetargeter,args)
    #s1='/home/chrisus/Proyectofinal/Motion/processed/S11/'
    #print(np.shape(np.load('/home/chrisus/Proyectofinal/Motion/processed/S11/Directions 1.54138969_2.npy')))
    a=np.load('/home/chrisus/Proyectofinal/Motion/disptec-2020/tools/motion_predictor/output/final/testing/input_aber00.npy')
    b=np.load('/home/chrisus/Proyectofinal/Motion/disptec-2020/tools/motion_predictor/output/final/testing/objt_ff.npy')
    c=np.load('/home/chrisus/Proyectofinal/Motion/disptec-2020/tools/motion_predictor/output/final/testing/finalobjt.npy')
    target=np.load('/home/chrisus/Proyectofinal/Motion/disptec-2020/tools/motion_predictor/output/target.npy')
    print(np.shape(a),'  ',np.shape(b), '' ,np.shape(c))
    
    trupred=np.zeros((1,64,15,2))
    bfpred=np.zeros((32,64,15,2))
    #print(a[0])
    #print('>?>>>>>?>>?>???>????>>>')
    '''
    samp,fram,_=np.shape(a)
    for n in range(samp):
        a[n]=np.interp(a[n], (a[n].min(), a[n].max()), (-300, +300))
    #print(a[0])
    #print('>?>>>>>?>>?>???>????>>>')
    samp,fram,_=np.shape(b)
    for n in range(samp):
        for fr in range(fram):
            b[n][fr]=np.interp(b[n][fr], (b[n][fr].min(), b[n][fr].max()), (64.5, +317.25))
    print(b[0][0])
    
    
    
    samp,fram,_=np.shape(c)
    for n in range(samp):
        for fr in range(fram):
            c[n][fr]=np.interp(c[n][fr], (c[n][fr].min(), c[n][fr].max()), (64.5, +317.25))
    
    
    for x in range(32):
        for f in range(64):
            for j in range(15):
                    #trupred[0][f][j][0]=-b[0][f][j*2]
                    #trupred[0][f][j][1]=-b[0][f][j*2+1]
                    bfpred[x][f][j][0]=-c[x][f][j*2]
                    bfpred[x][f][j][1]=-c[x][f][j*2+1]
        bfpred[x][:(32+x)]=a[:(32+x)]'''
    #trupred[0][:32]=a[:32]
    #print(bfpred[31]-a)                
    #print(trupred[0][32])
    #print(trupred[0][33])
    #print(b[0])
    #print(';;;;;;;;;;;;;;;;')
    #print(c[0])
    #np.save('/home/chrisus/Proyectofinal/Motion/disptec-2020/tools/motion_predictor/output/final/testing/tru_00.npy',trupred)
    #np.save('/home/chrisus/Proyectofinal/Motion/disptec-2020/tools/motion_predictor/output/final/testing/objt_ff.npy',bfpred)
    #samples,frames,joint2=np.shape(b)
    #a=[a]
    #print(a[0]-c[0][0])
    #print(np.shape(a))
    d=np.load('/home/chrisus/Desktop/Panoptics testing/unet_test.npy')
    def concatenate_outs_data():
        pose_files='/home/chrisus/Proyectofinal/Motion/disptec-2020/tools/motion_predictor/output/'
        pose_file=['bacth1qdrc.npy','bacth2qdrc.npy','bacth3qdrc.npy','bacth4qdrc.npy','bacth5qdrc.npy','bacth6qdrc.npy','bacth7qdrc.npy','bacth8qdrc.npy','bacth9qdrc.npy','bacth10qdrc.npy','bacth11qdrc.npy','bacth12qdrc.npy',]
           
        for p in range(len(pose_file)):
            print(pose_file[p][0])
            if pose_file[p][0]=='b':
                print(pose_file[p])
                out = np.load(pose_files+pose_file[p])
                print('out att',np.shape(out))
                if p!=0:
                    f=np.concatenate((f,out),axis=0)
                else:
                    f=out


        #print('atte concatenate',np.shape(f))
        #np.save(pose_files+'qdrcinput.npy',f)
        return f

    def process_data_for_aberman(data):
        samples,frames,joint2=np.shape(data)
        d=np.zeros((samples,frames,15,2))
        for n in range(samples):
            for f in range(frames):
                for j in range(int(round(joint2/2))):
                    d[n][f][j][0]=b[n][f][j*2]
                    d[n][f][j][1]=b[n][f][j*2+1]
        print(np.shape(d))
        return d
            
    #b=
    #b=process_data_for_aberman(b)
    #c=process_data_for_aberman(c)
    #o='output/'
    def data_on_aberman(a):
        pose_files='/home/chrisus/Proyectofinal/Motion/disptec-2020/tools/motion_predictor/output/final/testing/'
        encoded_s='/home/chrisus/Proyectofinal/Motion/disptec-2020/tools/motion_predictor/output/final/testing/'
        encoded_v='/home/chrisus/Proyectofinal/Motion/disptec-2020/tools/motion_predictor/output/final/testing/'
        ns,fps,jns,xyd=np.shape(a)
        target=np.zeros((ns,fps,jns,xyd))
        for j in range(ns):
            
            v, m1, b2, v3=TestMotionRetargeter.retarget_motion(TestMotionRetargeter,a[j],a[j],a[j])
            target[j]=v
            
            torch.save(b2, encoded_s+'sf'+str(j)+'.pt')
            torch.save(v3, encoded_v+'vf'+str(j)+'.pt')
        np.save(pose_files+'target_final_abermanff.npy',target)
        print(np.shape(target))

    #
    
	
    rfsk=['96','37','14','61','72','84','45','75','49','23',
    '73','17','79','93','7','6','69','55','51','9','18','8',
    '32','89','70','78','58','27','43','42','94','74','86','35',
    '62','54','97','85','50','15','90','76','16','22','91','26','57','80',
    '64','82','56','21','87','47','48','5','28','44','4','13','29','68','71',
    '66','20','39','99','41','36','81','38','98','31','83','95','53',
    '67','46','12','25','52','10','65','92','34','63','3','40','60','19',
    '24','33','59','77','30','11','0','2','1','88']
    '''
    
    
    pose_files='/home/chrisus/Proyectofinal/Motion/disptec-2020/tools/motion_predictor/output/final/testing/'
    retarget=np.zeros((32,15,2))
    retarget=target[0][:32]
    m32 = TestMotionRetargeter.encode_motion(TestMotionRetargeter,retarget)
    t32=m32.detach().cpu().numpy()
    np.save(pose_files+'input_aberman_test_for_att_unet_.npy',t32)
    '''
    #data_on_aberman([target[0]])
    #print(b[0])
    #save_dir='/home/chrisus/Proyectofinal/Motion/disptec-2020/tools/motion_predictor/output/final/testing/fffff.npy'
    save_dir='/home/chrisus/Desktop/Panoptics testing/out_unet_aberman.npy'
    v=TestMotionRetargeter.generate_encoded_files(TestMotionRetargeter,d,save_dir)
    #save_dir='/home/chrisus/Proyectofinal/Motion/disptec-2020/tools/motion_predictor/output/final/bffinal02.npy'
    #v=TestMotionRetargeter.generate_encoded_files(TestMotionRetargeter,c,save_dir)
    #-c /home/chrisus/Proyectofinal/Motion/disptec-2020/tools/motion_predictor/
    #--model_path /home/chrisus/Proyectofinal/Motion/disptec-2020/data/correction/pretrained_full.pth
    #python3 train.py -c /home/chrisus/Proyectofinal/Motion/disptec-2020/tools/motion_predictor/config.txt --model_path /home/chrisus/Proyectofinal/Motion/disptec-2020/data/correction/pretrained_full.pth -n full -g 0

if __name__ == '__main__':
    main()