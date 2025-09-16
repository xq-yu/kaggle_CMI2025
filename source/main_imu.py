import pandas as pd
import numpy as np
import torch
from sklearn.metrics import f1_score
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial.transform import Rotation as R
import warnings
import time
import os
import random
import sys
from util.metric import compute_metric
from data.dataset import create_dataset
import logging
import json
from config import Config_imu as Config
warnings.filterwarnings("ignore")

from train.train import train
from model.branchnet import RNNClassifier

#####################################  Config Setting  ###########################################
args = sys.argv

if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
    # Set device to cuda:1 (second GPU)
    device = torch.device('cuda:1')
    print(f"Found 2 GPUs. Using GPU 1: {torch.cuda.get_device_name(1)}")
elif torch.cuda.is_available() and torch.cuda.device_count() == 1:
    device = torch.device('cuda:0')
    print(f"Found 1 GPUs. Using GPU 0: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device('cpu')

if len(args)>1:
    training_seed = int(args[1])
else:
    training_seed = 1

Config.training_seed=1

if not os.path.exists(Config.output_dir):
    os.makedirs(Config.output_dir)  # 创建单级目录


if not os.path.exists(Config.train_config['model_dir']):
    os.makedirs(Config.train_config['model_dir'])  # 创建单级目录
    
    
#####################################  Utilities  ###########################################
def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def save_config(config, path):
    """
    Save configuration parameters to a JSON file.

    This function serializes a configuration object to JSON format, excluding
    internal Python attributes. It can handle both absolute and relative paths,
    automatically appending .json extension if not present.

    Args:
        config (Config): Configuration object to save.
        path (str): Path where the configuration should be saved.
    """
    dic = config.__dict__.copy()
    del (dic["__doc__"], dic["__module__"], dic["__dict__"], dic["__weakref__"])

    if not path.endswith(".json"):
        path += ".json"

    with open(path, "w") as f:
        json.dump(dic, f)

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{Config.train_config["model_dir"]}logs.txt'),  # 输出到文件
        logging.StreamHandler()          # 输出到控制台
    ]
)




if __name__ == "__main__":
    start_time = time.time()
    save_config(Config,f'{Config.train_config["model_dir"]}config.json')
        
    seed_everything(seed=Config.training_seed)
    dataset_train,_ = create_dataset( data_path=Config.data_dir,
                                                fold_file = Config.folds_info_file,
                                                fold_id = None,
                                                imu_only=Config.train_config['imu_only'],
                                                aug_config=Config.aug_config,
                                                )
    
    # dataset.save_scaler(Config.output_dir+'scaler.pkl')
    # dataset.save_fllna(Config.output_dir+'fillna_value.pkl')

    # with open(Config.output_dir+'scaler.pkl','rb') as f:
    #     scaler = pickle.load(f)
    # with open(Config.output_dir+'fillna_value.pkl','rb') as f:
    #     fillna_value = pickle.load(f)

    fea_std = torch.FloatTensor(dataset_train.cols_std)
    fea_mean = torch.FloatTensor(dataset_train.cols_mean)

    # define_model:

    
    score_tmp = []
    folds_f1,folds_logits,folds_seq,folds_epoch = [],[],[],[]
    for fold_id in range(Config.fold_num):
        logging.info(f'-------------Training Fold {fold_id}-------------')
        model = RNNClassifier(imu_only=Config.imu_only,
                              imu_dim=Config.fea_config['imu_dim'],
                              thmtof_dim=Config.fea_config['thmtof_dim'],
                              n_classes=Config.train_config['num_class'],
                              fea_std=fea_std,
                              fea_mean=fea_mean).to(device)
        
        dataset_train,dataset_val = create_dataset(data_path=Config.data_dir,
                                                   fold_file = Config.folds_info_file,
                                                   fold_id = fold_id,
                                                   imu_only=Config.train_config['imu_only'],
                                                   aug_config=Config.aug_config
                                                   )

        f1,epoch,oof_logits,sequence_id = train(model=model,
                                                dataset_train=dataset_train,
                                                dataset_val=dataset_val,
                                                fold_id=fold_id,
                                                config=Config.train_config,
                                                log=True,
                                                device=device)
        
        
        folds_f1.append(f1)
        folds_logits.append(oof_logits)
        folds_seq.append(sequence_id)
        folds_epoch.append(epoch)
    folds_logits = np.concatenate(folds_logits,axis=0)
    folds_seq = np.concatenate(folds_seq)
    
    oof_pred = pd.DataFrame(folds_logits,columns=Config.gesture_list)
    oof_pred['sequence_id'] = folds_seq
    oof_pred.to_csv(Config.train_config['model_dir']+'oof_pred.csv',index=False)

    sample = pd.read_csv(Config.data_dir+'sample.csv')
    sample = sample.loc[~sample.subject.isin(['SUBJ_019262','SUBJ_045235']),:]
    oof = pd.merge(sample,oof_pred,on='sequence_id',how='left')
    oof_pred = np.argmax(oof[Config.gesture_list].values,axis=1)

    oof_f1,binary_f1,macro_f1 = compute_metric(oof.y,oof_pred)

    tmp = "|".join([f'{i:0.4}' for i in folds_f1])
    logging.info(f'f1:{tmp}  |overall_F1 {oof_f1:.4f}')
    tmp = '|'.join([str(i) for i in folds_epoch])
    logging.info(f'epoch:{tmp}  |avg_epoch {int(np.mean(folds_epoch))}')

    logging.info('------------------Training Finished------------------')



    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time  # 计算耗时
    logging.info(f"程序运行耗时: {elapsed_time/60:.4f} 分")