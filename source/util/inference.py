import json
import torch
from torch import nn
import numpy as np
import pandas as pd
import pickle
from model.branchnet import RNNClassifier
from data.dataset import SensorDataset
model_dir = '../model/only_IMU/seed_1/'
def get_config(file):
    with open(file) as f:
        return json.load(f)

def load_model(model_dir):
    '''
    load folds model from one direction, the name of weights should be model_weight_fold{i}
    there should be a config.json in the same direction
    '''

    config = get_config(model_dir+'config.json')

    model_list = nn.ModuleList()
    for i in range(config['fold_num']):
        model = RNNClassifier(imu_only= config['train_config']['imu_only'],
                              imu_dim=config['fea_config']['imu_dim'],
                              n_classes=config['train_config']['num_class'],
                              thmtof_dim = config['fea_config']['thmtof_dim']
                              )
        
        model.load_state_dict(torch.load(f'{model_dir}/model_weight_fold{i}.pth'))
        model.eval()
        model_list.append(model)
    return model_list,config['train_config']['imu_only']


class FoldsBaggingModel(nn.Module):
    def __init__(self, model_dir):
        super().__init__()
        self.models,self.imu_only = load_model(model_dir)

    def __call__(self,X) ->np.ndarray:
        outputs = [model(X)[0] for model in self.models]
        logits = torch.mean(torch.stack(outputs), dim=0).cpu().detach().numpy()
        return logits
    

class Classifier():
    def __init__(self,model_dir_list,device,seq_len=128,imu_weight=0.5,all_weight=0.5):
        self.model_dir_list = model_dir_list
        self.device = device
        self.seq_len = 128

        self.imu_weight = imu_weight
        self.all_weight = all_weight

        self.imu_only_models = []
        self.imu_only_weights = []
        self.all_sensor_models = []
        self.all_sensor_weights = []

        self.imu_only_weight = 1
        self.all_sensor_weight = 1
        
        for model_dir in model_dir_list:
            model =  FoldsBaggingModel(model_dir).to(device)
            if model.imu_only:
                self.imu_only_models.append(model)
            else:
                self.all_sensor_models.append(model)

        self.gesture_list = [
        "Above ear - pull hair",
        "Cheek - pinch skin",
        "Eyebrow - pull hair",
        "Eyelash - pull hair",
        "Forehead - pull hairline",
        "Forehead - scratch",
        "Neck - pinch skin",
        "Neck - scratch",
        
        "Drink from bottle/cup",
        "Feel around in tray and pull out an object",
        "Glasses on/off",
        "Pinch knee/leg skin",
        "Pull air toward your face",
        "Scratch knee/leg skin",
        "Text on phone",
        "Wave hello",
        "Write name in air",
        "Write name on leg"
        ]
        
    def predict_imu(self,df):
        X=SensorDataset(
            df.copy(),
            data_aug=False,
            aug_config=None,
            imu_only=True,
            seq_len=self.seq_len,
            training=False,
            on_fly=False
        ).sequences
        X = torch.FloatTensor(X).to(self.device)

        logits = np.array([0.0]*18)
        weights = [1/len(self.imu_only_models) for i in range(len(self.imu_only_models))]
        
        for model,w in zip(self.imu_only_models,weights):
            logits+=w*model(X)
        return logits
    
    def predict_all(self,df):
        X=SensorDataset(
            df.copy(),
            data_aug=False,
            aug_config=None,
            imu_only=False,
            seq_len=self.seq_len,
            training=False,
            on_fly=False
        ).sequences
        X = torch.FloatTensor(X).to(self.device)

        logits = np.array([0.0]*18)
        weights = [1/len(self.all_sensor_models) for i in range(len(self.all_sensor_models))]
        for model,w in zip(self.all_sensor_models,weights):
            logits+=w*model(X)
        return logits
    

    def predict(self,sequence:pd.DataFrame,demo:pd.DataFrame,batch=False):
        df = sequence
        df['handedness'] = demo['handedness'].iloc[0]

        imu_flg = df[[f'tof_{i}_v{j}' for i in range(1,6) for j in range(64)]].isnull().values.mean() > 0.2
        if imu_flg:
            logits = self.predict_imu(df)
        else:
            logits = self.imu_weight*self.predict_imu(df)+self.all_weight*self.predict_all(df)

        idx = int(logits.argmax())
        gesture = self.gesture_list[idx]
        return logits,gesture