import pandas as pd
import numpy as np
import random
from itertools import product
import warnings
from util.aug_func import stretch_sequence
from util.quat_calulation import (relative_rotation_quaternion,
                                  sep_gravity,
                                  quaternion_to_euler,
                                  quaternion_multiply,
                                  euler_to_quaternion,
                                  quaternion_conjugate,
                                  )
from scipy.spatial.transform import Rotation as R
import time
warnings.filterwarnings("ignore")

def seq_cutoff(df,seq_len):
    '''
    padding cut sequences to a fixed length
    df: [sequence_id,sequence_counter,.....]
    '''
    combinations = list(product(df.sequence_id.unique(), list(range(seq_len))))
    df['sequence_counter'] = df[['sequence_id']].groupby('sequence_id')['sequence_id'].cumcount()
    df_new = pd.DataFrame(combinations, columns=['sequence_id', 'sequence_counter']).sort_values(['sequence_id', 'sequence_counter'])
    
    df['sequence_counter'] = df['sequence_counter'] - (df[['sequence_id','sequence_counter']].groupby('sequence_id')['sequence_counter'].transform('max')-seq_len)
    df_new = pd.merge(df_new,df,how='left',on=['sequence_id','sequence_counter']).fillna(0)

    return df_new


def quat_diff(df:pd.DataFrame,delta:int):
    '''
    calculate the relative rotation quaternion between two time steps
    Arguments:
        df: pd.DataFrame of quat [seq_id,w,x,y,z]
        delta: int, time step shift steps
    return:
        quat_rel: [N,[w,x,y,z]]
    '''
    #df = df.fillna(pd.Series({'rot_x':-0.119916,'rot_y':-0.059953,'rot_z':-0.188298,'rot_w':0.360375}))
    quat_shift = np.array(df.groupby('sequence_id').shift(delta).bfill())
    quat = np.array(df[['rot_w','rot_x','rot_y','rot_z']])

    quat_rel = relative_rotation_quaternion(quat,quat_shift,local=True)
    quat_rel = quat_rel*((quat_rel[:,0:1]>0)*2-1)
    return quat_rel

def Fix_outlier(df,imu_only):
    '''
    Fix data of the uncorrectly worn IMU sensors
    Arguments:
        df: pd.DataFrame of the whole data
        imu_only: bool, whether only process imu data
    return:
        df: pd.DataFrame of the processed data
    '''

    outlier_idx=df.subject.isin(['SUBJ_019262','SUBJ_045235'])
    df_outlier = df.loc[outlier_idx,:]
    quat = df_outlier[['rot_x','rot_y','rot_z','rot_w']].fillna(pd.Series({'rot_x':-0.119916,'rot_y':-0.059953,'rot_z':-0.188298,'rot_w':0.360375}))

    # rotate 180 degree around local z axis
    d = 180
    quat_rot_z = np.array([np.cos(np.deg2rad(d)/2),0,0,np.sin(np.deg2rad(d)/2)])
    quat_rot = quaternion_multiply(quat[['rot_w','rot_x','rot_y','rot_z']].values,quat_rot_z)
    quat_rot = quat_rot*((quat_rot[:,0:1]>0)*2-1)
    df.loc[outlier_idx,['rot_w','rot_x','rot_y','rot_z']]=quat_rot
    df.loc[outlier_idx,'acc_x'] *= -1
    df.loc[outlier_idx,'acc_y'] *= -1

    if not imu_only:
        tof_columns = [f'tof_{i}_v{j}' for i in range(1,6) for j in range(64)]
        thm_columns = [f'thm_{i}' for i in range(1,6)]
        tof = df_outlier[tof_columns].values.reshape(-1,5,8,8)
        tof = np.rot90(tof, k=2, axes=(2, 3))
        tof[:,[0,1,2,3,4]] = tof[:,[0,3,4,1,2]]
        tof = tof.reshape(-1,5*64)

        thm = df_outlier[thm_columns].values
        thm[:,[0,1,2,3,4]] =  thm[:,[0,3,4,1,2]]
        df.loc[outlier_idx,tof_columns] = tof
        df.loc[outlier_idx,thm_columns] = thm
        
    return df

def lefthanded_flip(df,imu_only):
    '''
    Flip the data for left-handed users
    Arguments:
        df: pd.DataFrame of the lefthanded data
        imu_only:whether only process imu data
    Return:
        df: pd.DataFrame of the processed data
    '''
    df = df.copy()
    df['acc_x'] = -df['acc_x']
    # 先整体旋转，保持和右撇子朝向一致
    d=-110
    quat_rot_z = np.array([np.cos(np.deg2rad(d)/2),0,0,np.sin(np.deg2rad(d)/2)])

    #quat = df[['rot_x','rot_y','rot_z','rot_w']].fillna(pd.Series({'rot_x':-0.119916,'rot_y':-0.059953,'rot_z':-0.188298,'rot_w':0.360375}))
    quat = df[['rot_x','rot_y','rot_z','rot_w']]
    
    quat_flip = quaternion_multiply(quat_rot_z, quat[['rot_w','rot_x','rot_y','rot_z']].values)
    quat_flip = pd.DataFrame(quat_flip,columns=['rot_w','rot_x','rot_y','rot_z'])
    # 计算欧拉角后翻转y和z的旋转
    euler_x, euler_y, euler_z = quaternion_to_euler(quat_flip[['rot_x','rot_y','rot_z','rot_w']].values)  #zyx  yaw pitch roll
    quat_flip = euler_to_quaternion(euler_x, -euler_y, -euler_z)
    quat_flip = quat_flip*((quat_flip[:,0:1]>0)*2-1)  # 保证w均为正
    df[['rot_w', 'rot_x', 'rot_y', 'rot_z']] = quat_flip
    
    if not imu_only:
        # 左撇子THM TOF数据翻转
        # tof1,2,4 左右翻转
        # tof3,5 上下翻转并交换传感器编号
        
        x = df[[f'tof_{i}_v{j}' for i in range(1,6) for j in range(64)]].values    
        x = x.reshape((len(x),5,8,8))
        x[:,[0,1,3]] =  x[:,[0,1,3],:,::-1]
        x[:,[2,4]] =  x[:,[2,4],::-1]
        x[:,[2,4]]= x[:,[4,2]]
        df[[f'tof_{i}_v{j}' for i in range(1,6) for j in range(64)]] = x.reshape((len(x),-1))
        df['thm_3'],df['thm_5'] = df['thm_5'].values,df['thm_3'].values
    return df


def quat_smooth(df):
    '''
    Smooth the quaternion data to avoid sudden jumps
    '''
    df = df[['sequence_id','rot_x','rot_y','rot_z','rot_w']].copy()
    df_shift = df[['sequence_id','rot_x','rot_y','rot_z','rot_w']].groupby('sequence_id').shift(1).fillna(method='bfill')
    flip_flg = ((np.abs(df.rot_x+df_shift.rot_x)+np.abs(df.rot_y+df_shift.rot_y)+np.abs(df.rot_z+df_shift.rot_z))<0.5) & (df.rot_w<0.1)
    df['flip_flg'] = flip_flg
    df['flip_flg'] = df.groupby('sequence_id')['flip_flg'].cumsum()
    df['rot_x'] = df.rot_x*((-1)**df.flip_flg)
    df['rot_y'] = df.rot_y*((-1)**df.flip_flg)
    df['rot_z'] = df.rot_z*((-1)**df.flip_flg)
    df['rot_w'] = df.rot_w*((-1)**df.flip_flg)
    return df


def DataProcess_imu(df,training):
    '''
    IMU data process and feature engineering
    '''

    imu_cols = ['acc_x','acc_y','acc_z','rot_w','rot_x','rot_y','rot_z']
    df = df[['sequence_id']+imu_cols]
    acc = np.array(df[['acc_x','acc_y','acc_z']])
    # remove gravity from raw acc
    if training:
        # quat = np.array(df[['rot_x','rot_y','rot_z','rot_w']].fillna(df[['rot_x','rot_y','rot_z','rot_w']].mean()))
        quat = np.array(df[['rot_x','rot_y','rot_z','rot_w']].fillna(pd.Series({'rot_x':-0.119916,'rot_y':-0.059953,'rot_z':-0.188298,'rot_w':0.360375})))
    else:
        quat = np.array(df[['rot_x','rot_y','rot_z','rot_w']].fillna(pd.Series({'rot_x':-0.119916,'rot_y':-0.059953,'rot_z':-0.188298,'rot_w':0.360375})))
    
    acc_new,gravity = sep_gravity(acc,quat)
    df[['acc_x_new','acc_y_new','acc_z_new']] = acc_new

    # 3. relative rotation quaternion
    quat_relative = quat_diff(df[['sequence_id','rot_w','rot_x','rot_y','rot_z']],1)
    df[['rot_delta1_w','rot_delta1_x','rot_delta1_y','rot_delta1_z']] = quat_relative
    quat_relative = quat_diff(df[['sequence_id','rot_w','rot_x','rot_y','rot_z']],2)
    df[['rot_delta2_w','rot_delta2_x','rot_delta2_y','rot_delta2_z']] = quat_relative

    df['euler_delta1_x'],df['euler_delta1_y'],df['euler_delta1_z'] = quaternion_to_euler(df[['rot_delta1_x','rot_delta1_y','rot_delta1_z','rot_delta1_w']].values)

    df['anguler_jerk_x'] = df.groupby('sequence_id')['euler_delta1_x'].diff().fillna(0)
    df['anguler_jerk_y'] = df.groupby('sequence_id')['euler_delta1_y'].diff().fillna(0)
    df['anguler_jerk_z'] = df.groupby('sequence_id')['euler_delta1_z'].diff().fillna(0)

    df['anguler_snap_x'] = df.groupby('sequence_id')['anguler_jerk_x'].diff().fillna(0)
    df['anguler_snap_y'] = df.groupby('sequence_id')['anguler_jerk_y'].diff().fillna(0)
    df['anguler_snap_z'] = df.groupby('sequence_id')['anguler_jerk_z'].diff().fillna(0)

    df['acc_jerk_x'] = df.groupby('sequence_id')['acc_x'].diff().fillna(0)
    df['acc_jerk_y'] = df.groupby('sequence_id')['acc_y'].diff().fillna(0)
    df['acc_jerk_z'] = df.groupby('sequence_id')['acc_z'].diff().fillna(0)

    df['acc_snap_x'] = df.groupby('sequence_id')['acc_jerk_x'].diff().fillna(0)
    df['acc_snap_y'] = df.groupby('sequence_id')['acc_jerk_y'].diff().fillna(0)
    df['acc_snap_z'] = df.groupby('sequence_id')['acc_jerk_z'].diff().fillna(0)
    
    df['acc_mag'] = np.sqrt(np.sum(acc_new**2,axis=1))
    df['rot_angle'] = 2 * np.arccos(df['rot_w'].clip(-1, 1))
    df['acc_mag_jerk'] = df.groupby('sequence_id')['acc_mag'].diff().fillna(0)
    df['rot_angle_vel'] = df.groupby('sequence_id')['rot_angle'].diff().fillna(0)

    return df[[col for col in df.columns if col not in imu_cols+['sequence_id']]]

def DataProcess_tof(df):
    tof_cols = [f'tof_{i}_v{j}' for i in range(1,6) for j in range(64)]
    df[tof_cols] = df[tof_cols].replace(-1,255)  
    return df

class DataAugmentation():
    def __init__(self,aug_config,on_fly):
        self.aug_config =  aug_config
        self.imu_cols = ['acc_x','acc_y','acc_z','rot_w','rot_x','rot_y','rot_z']
        self.thm_cols = [f'thm_{i}' for i in range(1,6)]
        self.tof_cols = [f'tof_{i}_v{j}' for i in range(1,6) for j in range(64)]
        self.ext_cols = ['phase']
        self.meta_cols = ['orientation',
                          'sequence_type',
                          'subject',
                          'handedness',
                          'sequence_id',
                          'gesture']

        self.on_fly = on_fly

    def __call__(self,df):
        df = df.copy()
        imu_columns = self.imu_cols
        # imu rotation
        imu_aug = self.imu_rotate_aug(df[['sequence_id']+imu_columns])
        df[imu_columns] = imu_aug[imu_columns]
        
        # cols stretch
        stretch_cols = [col for col in  df.columns if col in self.imu_cols+self.thm_cols+self.tof_cols]
        df_strech = self.time_stretch_aug(df[['sequence_id']+stretch_cols])
        rest_cols = [col for col in  df.columns if col not in stretch_cols]
        df_rest = df[rest_cols].drop_duplicates(['sequence_id'],keep='first')
        df = pd.merge(df_strech,df_rest,on='sequence_id',how='left')

        # shift aug
        df = self.time_shift_aug(df)

        # imu drop aug
        df = self.imu_drop_aug(df)

        # start_time = time.time()
        # # top drop aug
        # if 'tof_1_v1' in df.columns:
        #     df = self.tof_drop_aug(df)
        # elapsed_time = time.time() - start_time
        # print(f"函数运行时间：{elapsed_time} 秒")
        
        return df
        


    def sequence_hash(self,sequence_id):
        salt = str(np.random.random)[0:10]
        return np.array([hash(i+salt) for i in sequence_id])
        

    def _time_stretch(self,sequence:np.ndarray,p)->np.ndarray:
        time_stretch_range = (0.8, 1.2)
        if np.random.random()<p:
            if np.random.random() < 0.5 or len(sequence) < 15:
                # Stretch entire sequence
                stretch_factor = random.uniform(*time_stretch_range)
                sequence = stretch_sequence(sequence, stretch_factor)
            else:
                start = random.randint(0, sequence.shape[0] - 10)
                end = random.randint(start + 5, sequence.shape[0] - 1)
                stretch_factor = random.uniform(*time_stretch_range)
                stretched = stretch_sequence(sequence[start:end], stretch_factor)
                sequence = np.concatenate([sequence[:start], stretched, sequence[end:]], axis=0)
        return sequence

    def time_stretch_aug(self,df):
        """
        Stretch/compress the sequence or part of it in time.
        """
        p = self.aug_config['stretch_prop']
        value_cols = [col for col in df.columns if col!='sequence_id']
        #if self.on_fly:
        df_aug = df.groupby('sequence_id').apply(lambda x:pd.DataFrame(self._time_stretch(x[value_cols].values,p),columns=value_cols))
        # else:
        #     df_aug = df.groupby('sequence_id').parallel_apply(lambda x:pd.DataFrame(self._time_stretch(x[value_cols].values,p),columns=value_cols))
        df_aug['sequence_id'] = df_aug.index.get_level_values(0)
        df_aug = df_aug.reset_index(drop=True)
        return df_aug


    def _time_shift(self,df:pd.DataFrame,p)->  pd.DataFrame:
        time_shift_range = 0.1
        max_shift = int(len(df) * time_shift_range)
        if np.random.random()<p:
            shift = random.randint(-max_shift, max_shift)
            df = df.shift(shift).dropna(how='all')
        return df

    def time_shift_aug(self,df) -> pd.DataFrame:
        """
        Shift the sequence in time.
        """
        df = df.copy()
        p = self.aug_config['shift_prop']
        df = df.groupby('sequence_id',as_index=False).apply(lambda x:self._time_shift(x,p)).reset_index(drop=True)
        return df 
    
    

    def imu_drop_aug(self, df:pd.DataFrame, thm=None, tof=None):
        df = df.copy()
        cols_init = df.columns
        p = self.aug_config['rot_drop_prop']
        df['seq_len'] = 0
        df['sequence_counter'] = df.groupby('sequence_id').cumcount()
        df['rot_drop'] = self.sequence_hash(df.sequence_id)%100/100<p
        df['seq_len'] = df.groupby('sequence_id')['sequence_id'].transform('count')

        salt = str(np.random.random())[0:10]
        df['rot_drop_start'] = [0 if seq_len<20 or np.random.random() <= 0.5 else hash(seq_id+salt)%(seq_len-10) for seq_len,seq_id in zip(df['seq_len'],df['sequence_id'])]
        df.loc[(df.sequence_counter>df.rot_drop_start) & (df.rot_drop),['rot_w,rot_x','rot_y','rot_z']] = np.nan
        df = df[cols_init]
        return df

    def _tof_drop(self,df:pd.DataFrame,p):
        df =df.copy()
        tof = df[self.tof_cols].values
        if np.random.random() <= p:
            # Drop pixels of a sensor
            n = np.random.randint(1, tof.shape[1] // 10)
            to_drop = np.random.choice(tof.shape[1], n)
            tof[:, to_drop] = 1

        if np.random.random() <= p:
            # Drop timesteps
            if tof.shape[0] > 40:
                n = np.random.randint(1, tof.shape[0] // 20)
                to_drop = np.random.choice(tof.shape[0], n)
                tof[to_drop] = 1

        elif np.random.random() <= p:
            # Drop sensors at a timestep
            tof = tof.reshape(-1, 5, 8, 8).reshape(-1, 8, 8)
            if tof.shape[0] > 100:
                n = np.random.randint(1, tof.shape[0] // 10)
                to_drop = np.random.choice(tof.shape[0], n)
                tof[to_drop] = 1

        # Communication error: drop a sensor (partially)
        if np.random.random() <= p:
            tof = tof.reshape(-1, 5, 8, 8)
            # n_times = np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2])
            n_times = 1
            for _ in range(n_times):
                to_drop = np.random.randint(0, 5)  # sensor to drop
                if np.random.random() <= (0.5 + (n_times - 1) * 0.1) or tof.shape[0] < 20:
                    start = 0  # drop from the beginning
                else:  # drop from a random timestep
                    start = np.random.randint(0, tof.shape[0] - 10)
                tof[start:, to_drop] = 1
        tof = tof.reshape(-1, 320)
        df[self.tof_cols] = tof
        return df


    def tof_drop_aug(self,df):
        df = df.copy()
        p = self.aug_config['tof_drop_prop']
        drop_idx = self.sequence_hash(df.sequence_id)%100/100<p
        df_aug = df.loc[drop_idx,['sequence_id']+self.tof_cols]

        #if self.on_fly:
        df_aug = df_aug.groupby('sequence_id',as_index=False).apply(lambda x:self._tof_drop(x,p)).reset_index(drop=True)
        #else:
        #df_aug = df_aug.groupby('sequence_id',as_index=False).parallel_apply(lambda x:self._tof_drop(x,p)).reset_index(drop=True)
        df[self.tof_cols] = df_aug[self.tof_cols]
        return df


    def imu_rotate_aug(self,df_imu):
        df = df_imu.copy()
        aug_prop = self.aug_config['imu_rotate_prop']
        aug_idx = [np.random.random()<aug_prop for _ in range(len(df.sequence_id.unique()))]

        deg_z = [np.random.choice(list(range(15,46))+list(range(-45,-14))) if i else 0 for i in aug_idx]
        deg_y = [np.random.choice(list(range(3,8))+list(range(-7,-2))) if i else 0 for i in aug_idx]

        # deg_z = [np.random.random()*120-60 if i else 0 for i in aug_idx]
        # deg_y = [np.random.random()*14-7 if i else 0 for i in aug_idx]

        quat_rot_z = pd.DataFrame({'sequence_id':df.sequence_id.unique()})
        quat_rot_z[['rot_w','rot_x','rot_y','rot_z']] = np.array([np.array([np.cos(np.deg2rad(d)/2),0,0,np.sin(np.deg2rad(d)/2)]) for d in deg_z])
        quat_rot_z = pd.merge(df[['sequence_id']],quat_rot_z,how='left',on='sequence_id')[['rot_w','rot_x','rot_y','rot_z']].values
        quat_rot_y = pd.DataFrame({'sequence_id':df.sequence_id.unique()})
        quat_rot_y[['rot_w','rot_x','rot_y','rot_z']] = np.array([np.array([np.cos(np.deg2rad(d) / 2), 0, np.sin(np.deg2rad(d) / 2), 0]) for d in deg_y])
        quat_rot_y = pd.merge(df[['sequence_id']],quat_rot_y,how='left',on='sequence_id')[['rot_w','rot_x','rot_y','rot_z']].values
        quat_rot_y_reverse = quaternion_conjugate(quat_rot_y)

        # quat rotation
        quat_aug = df[['rot_w','rot_x','rot_y','rot_z']].values
        quat_aug = quaternion_multiply(quat_rot_z,quat_aug)  #[w,x,y,z]
        quat_aug = quaternion_multiply(quat_aug,quat_rot_y_reverse)
        quat_aug = quat_aug*((quat_aug[:,0:1]>0)*2-1)  # 保证w均为正

        # acc rotation
        acc_aug = df[['acc_x','acc_y','acc_z']].values
        rot = R.from_quat(quat_rot_y[:,[1,2,3,0]])
        acc_aug = rot.apply(acc_aug)

        df[['rot_w','rot_x','rot_y','rot_z']] = quat_aug
        df[['acc_x','acc_y','acc_z']] = acc_aug
        
        return df