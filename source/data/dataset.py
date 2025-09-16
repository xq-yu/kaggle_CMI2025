import pandas as pd
import numpy as np
from data.data_process import DataAugmentation
from torch.utils.data import Dataset
import torch
import pickle
from data.data_process import (DataProcess_imu,
                               DataProcess_tof,
                               Fix_outlier,
                               lefthanded_flip,
                               quat_smooth,
                               seq_cutoff)

import time


def file_split(data_path,fold_id,folds_file,used_columns,):
    df = pd.read_csv(f'{data_path}train.csv')
    demo = pd.read_csv(f'{data_path}train_demographics.csv')
    df = pd.merge(df,demo,how='left',on='subject')

    folds_info = pd.read_csv(folds_file)
    sequence_train =  list(folds_info.loc[folds_info[f'5fold_seed1']!=fold_id,'sequence_id'])
    sequence_val =  list(folds_info.loc[folds_info[f'5fold_seed1']==fold_id,'sequence_id'])

    df_train = df.loc[df.sequence_id.isin(sequence_train),used_columns]
    df_val = df.loc[df.sequence_id.isin(sequence_val),used_columns]
    return df_train.reset_index(drop=True),df_val.reset_index(drop=True)


def create_dataset(data_path,fold_file,fold_id,imu_only,aug_config,on_fly=False):
    used_columns = ['orientation','sequence_type','subject','handedness','sequence_counter','sequence_id','gesture']+['acc_x', 'acc_y', 'acc_z'] + ['rot_w', 'rot_x', 'rot_y', 'rot_z']
    if not imu_only:
        used_columns += [f'thm_{i}' for i in range(1,6)] + [f'tof_{i}_v{j}' for i in range(1,6) for j in range(64)]

    if fold_id is not None:
        df_train,df_val = file_split(data_path,fold_id,fold_file,used_columns)
        df_val = df_val.loc[~df_val.subject.isin(['SUBJ_019262','SUBJ_045235']),:]
        dataset_train = SensorDataset(df_train,
                                        data_aug=aug_config['if_aug'],
                                        aug_config = aug_config,
                                        imu_only=imu_only,
                                        training=True,
                                        on_fly = on_fly)
        dataset_val= SensorDataset(df_val,
                                    data_aug=False,
                                    aug_config = aug_config,
                                    imu_only=imu_only,
                                    training=False,
                                    on_fly = on_fly)
        return dataset_train,dataset_val
    else:
        df_train = pd.read_csv(f"{data_path}train.csv")
        demo = pd.read_csv(f'{data_path}train_demographics.csv')
        df_train = pd.merge(df_train,demo,how='left',on='subject')[used_columns]
        dataset_train = SensorDataset(df_train,
                                    data_aug=aug_config['if_aug'],
                                    aug_config = aug_config,
                                    imu_only=imu_only,
                                    training=True,
                                     on_fly = on_fly)
        return dataset_train,None


class SensorDataset(Dataset):
    def __init__(self,df,data_aug,aug_config,imu_only,fillna_value=None,scaler=None,seq_len=128,training=True,on_fly=False):
        
        # config
        self.seq_len = seq_len
        self.data_aug = data_aug
        self.aug_config = aug_config
        self.fillna_value = fillna_value
        self.scaler = scaler
        self.training = training
        self.seq_len = seq_len
        self.imu_only = imu_only
        self.on_fly = on_fly

        # data clean
        if training:
            df = Fix_outlier(df,imu_only)


        df_left = df.loc[df.handedness==0,:]
        if len(df_left)>0:
            df.loc[df.handedness==0,:] = lefthanded_flip(df_left,imu_only).values

        if data_aug:
            self.data_augmentation = DataAugmentation(self.aug_config,self.on_fly)

        self.df = df

        self.sequence_list = df.sequence_id.unique()

        # 1. label encoding
        self.label_index = {
        "Above ear - pull hair": 0,
        "Cheek - pinch skin": 1,
        "Eyebrow - pull hair": 2,
        "Eyelash - pull hair": 3, 
        "Forehead - pull hairline": 4,
        "Forehead - scratch": 5,
        "Neck - pinch skin": 6, 
        "Neck - scratch": 7,
        
        "Drink from bottle/cup": 8,
        "Feel around in tray and pull out an object": 9,
        "Glasses on/off": 10,
        "Pinch knee/leg skin": 11, 
        "Pull air toward your face": 12,
        "Scratch knee/leg skin": 13,
        "Text on phone": 14,
        "Wave hello": 15,
        "Write name in air": 16,
        "Write name on leg": 17
        }

        if not on_fly:
            self.update(idx=None)
            self.cols_std = np.std(self.sequences.reshape(-1,len(self.feature_cols)),axis=0)
            self.cols_mean = np.mean(self.sequences.reshape(-1,len(self.feature_cols)),axis=0)
            if not imu_only:
                self.cols_std[-320:] = 84
                self.cols_mean[-320:] = 187

    def update(self,idx=None):
        if idx is None:
            df = self.df.copy()
        else:
            df = self.df.loc[self.df.sequence_id==self.sequence_list[idx]]
        # data aug
        if self.data_aug:
            df =  self.data_augmentation(df)
        
        # data smooth
        df_smooth = quat_smooth(df[['sequence_id','rot_x','rot_y','rot_z','rot_w']])
        df[['rot_x','rot_y','rot_z','rot_w']] = df_smooth[['rot_x','rot_y','rot_z','rot_w']]

        # feature extend
        df_imu_ext = DataProcess_imu(df[['sequence_id','acc_x','acc_y','acc_z','rot_w','rot_x','rot_y','rot_z']],self.training)
        df = pd.concat([df,df_imu_ext],axis=1)
        if not self.imu_only:
            df=DataProcess_tof(df)

        # feature select
        self.feature_cols = (['acc_x', 'acc_y', 'acc_z'] +                                             
            ['rot_w', 'rot_x', 'rot_y', 'rot_z'] +                                     
            ['acc_x_new', 'acc_y_new', 'acc_z_new']+                          
            ['rot_delta1_w', 'rot_delta1_x', 'rot_delta1_y','rot_delta1_z']+
            ['acc_mag','rot_angle','acc_mag_jerk','rot_angle_vel']+
            ['rot_delta2_w', 'rot_delta2_x', 'rot_delta2_y','rot_delta2_z']+
            ['anguler_jerk_x','anguler_jerk_y','anguler_jerk_z']+
            ['anguler_snap_x','anguler_snap_y','anguler_snap_z']+
            ['acc_jerk_x','acc_jerk_y','acc_jerk_z']+
            ['acc_snap_x','acc_snap_y','acc_snap_z'])

        if not self.imu_only:
            self.feature_cols += (
                [f'thm_{i}' for i in range(1,6)] +
                [f'tof_{i}_v{j}' for i in range(1,6) for j in range(64)]
            ) 

        # fillna
        df = df.fillna(0)
        
        if 'gesture' not in df.columns:
            df['label']=-1
        else:
            df['label'] = df['gesture'].map(self.label_index)

        # create sample
        self.labels = np.array(df[['sequence_id','label']].drop_duplicates().sort_values(['sequence_id'])['label'].astype('int'))
        df = seq_cutoff(df,seq_len=self.seq_len)  #序列填充截断
        df = df.sort_values(['sequence_id','sequence_counter'])
        self.sequences = df[self.feature_cols].values.reshape(-1, self.seq_len, len(self.feature_cols))
        self.sequence_id=df.sequence_id.unique()

    def save_scaler(self,save_file):
        with open(save_file,'wb') as f:
            pickle.dump(self.scaler,f) 

    def save_fllna(self,save_file):
        with open(save_file,'wb') as f:
            pickle.dump(self.fillna_value,f) 

    def __len__(self):
        return len(self.sequence_list)
    
    def __getitem__(self, idx):
        if not self.on_fly:
            return self.sequence_id[idx],torch.FloatTensor(self.sequences[idx]), torch.LongTensor([self.labels[idx]])[0]
        else:
            self.update(idx)
            return self.sequence_id[0],torch.FloatTensor(self.sequences[0]), torch.LongTensor([self.labels[0]])[0]



