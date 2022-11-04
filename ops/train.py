from glob import glob
from importlib.resources import path
from pickle import TRUE
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

    def generate_decoded_files(s, pose_files, output_dir,before):
        print('generate encoded files')
        dir_name32='path_of_view_coded_data'
        dir_name64='path_of_skeleton_coded_data'
        suffix='.npy'
        sample,fps,joint,xy=np.shape(pose_files)
        print('samples',sample)
        rf=list(range(0, sample))
        rfsk=list(map(str,rf))
        print(rfsk)
        result=np.zeros(( sample,fps,joint,xy))
        print('final result shape',np.shape(result))
        for i in range(sample):

            b2=torch.load(dir_name64+'s'+rfsk[i]+'.pt')
            v3=torch.load(dir_name32+'v'+rfsk[i]+'.pt')
            if before==TRUE:
                m1 = s.encode_motion(s,pose_files[i])
            else:
                m1=torch.Tensor(pose_files[i])
            out = s.net.decoder(torch.cat([m1.cuda(), b2.cuda(), v3.cuda()], dim=1))
            
            out = denormalize_and_move_poses_to_numpy(out, s.mean_pose, s.std_pose, s.w_skeleton//2, s.h_skeleton//2)

            out = translate_poses_from_motion_retargeter(out)
            result[i]=out
        print(np.shape(result))
        np.save(output_dir,result)


    def generate_encoded_files(s, pose_files, output_dir):
        dir_name32='path_of_view_coded_data'
        dir_name64='path_of_skeleton_coded_data'
        suffix='.npy'
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



def main():
    # --model_path /home/chrisus/Proyectofinal/Motion/disptec-2020/data/correction/pretrained_full.pth -n full -g 0

    T=TestMotionRetargeter.__init__(TestMotionRetargeter,args)
    target_data=np.load('DATA_dir_target')
    predicted_data=np.load('path_predicted_data')
    #process_data_for_aberman(target_data) #data required shape (samples,64 or 32,15,2)
    


    def process_data_for_aberman(data):
        samples,frames,joint2=np.shape(data)
        d=np.zeros((samples,frames,15,2))
        for n in range(samples):
            for f in range(frames):
                for j in range(int(round(joint2/2))):
                    d[n][f][j][0]=data[n][f][j*2]
                    d[n][f][j][1]=data[n][f][j*2+1]
        print(np.shape(d))
        return d
            
    def data_on_aberman(a):
        pose_files='path_aberman_decoded_target'
        encoded_s='path_encoded_skeleton'
        encoded_v='path_encoded_View'
        ns,fps,jns,xyd=np.shape(a)
        target=np.zeros((ns,fps,jns,xyd))
        for j in range(ns):
            
            v, m1, b2, v3=TestMotionRetargeter.retarget_motion(TestMotionRetargeter,a[j],a[j],a[j])
            target[j]=v
            
            torch.save(b2, encoded_s+'sf'+str(j)+'.pt')
            torch.save(v3, encoded_v+'vf'+str(j)+'.pt')
        np.save(pose_files+'target_final_aberman.npy',target)
        print(np.shape(target))

    
    
    data_on_aberman(target_data)
    save_dir_coded_data='path_to_save_coded_data_for_training'
    generate_coded_input_data=TestMotionRetargeter.generate_encoded_files(TestMotionRetargeter,target_data,save_dir_coded_data)
    save_dir='path_to_save'
    before='False' #False if using predicted data before Aberman, True if using prediction of coded data
    v=TestMotionRetargeter.generate_decoded_files(TestMotionRetargeter,save_dir,before)
    
if __name__ == '__main__':
    main()