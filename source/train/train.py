import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings
import os
import random
from util.metric import compute_metric
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
import time
import copy
import logging
warnings.filterwarnings("ignore")


def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def validate(model,val_loader,criterion,device):
    model.eval()
    all_preds, all_labels,all_logits,sequence_id= [], [],[],[]
    with torch.no_grad():
        loss_avg = []
        for batch_sequence_id,batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss_avg.append(loss.item())
            _, preds = torch.max(logits, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
            all_logits.append(logits.cpu().numpy())
            sequence_id.extend(batch_sequence_id)

    
    all_logits=np.concatenate(all_logits,axis=0)
    f1,_,_ = compute_metric(np.array(all_labels), np.array(all_preds))
    loss_avg = np.mean(loss_avg)
    return f1,loss_avg,all_logits,sequence_id


class LabelSmoothingLoss(nn.Module):
    def __init__(self, epsilon=0.1):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        confidence = 1 - self.epsilon
        low_confidence = self.epsilon / (num_classes - 1)
        
        smoothed_targets = torch.full_like(logits, low_confidence)
        smoothed_targets.scatter_(1, targets.unsqueeze(1), confidence)

        loss = -torch.sum(smoothed_targets * F.log_softmax(logits, dim=1), dim=1)
        return loss.mean()


class ModelEMA:
    def __init__(self, model, decay=0.99):
        self.ema_model = copy.deepcopy(model).eval()
        self.decay = decay
        self.ema_model.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            msd = model.state_dict()
            for k, ema_v in self.ema_model.state_dict().items():
                model_v = msd[k].detach()
                if model_v.dtype.is_floating_point:
                    ema_v.copy_(ema_v * self.decay + (1. - self.decay) * model_v)
                else:
                    ema_v.copy_(model_v)



def train(model,dataset_train,dataset_val,fold_id,config,log,device):
    num_class = config['num_class']
    num_workers = config.get('num_workers',0)
    model_dir = config['model_dir']
    
    EMA = config['EMA']
    if EMA:
        ema = ModelEMA(model, decay=0.99)
    
    batch_size = config['batch_size']
    early_stop = config['early_stop']
    if early_stop and fold_id is None:
        raise ValueError(f'fold_id is None and early_stop cant be used!')

    patient = config['patient'] if early_stop else None
    epoch_num = config['epoch_num']
    lr = config['lr']
    warmup_p = config["warmup_prop"]
    weight_decay = config['weight_decay']
    
    fulldata_train = fold_id is None
    tag = str(fold_id) if fold_id is not None else 'all'
    

    # create dataloader
    train_loader = DataLoader(dataset_train, batch_size=batch_size,shuffle=True,drop_last=True,num_workers=num_workers)
    if fold_id is not None:
        val_loader = DataLoader(dataset_val, batch_size=batch_size,drop_last=False,num_workers=num_workers)

    #optimizer and criterion
    #优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = LabelSmoothingLoss(epsilon=0.1)  # 标签平滑损失

    # 学习率调度器
    num_training_steps = epoch_num * len(train_loader)
    num_warmup_steps = int(warmup_p * num_training_steps)


    if not early_stop:
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps
        )

    
    # 3. 训练循环
    best_f1=0
    best_epoch = 0
    for epoch in range(epoch_num):
        start_time = time.time()
        if not dataset_train.on_fly and dataset_train.data_aug:   # 如果不进行数据增强或者aug只在load时执行则不进行更新
            dataset_train.update()
            train_loader = DataLoader(dataset_train, batch_size=batch_size,shuffle=True,drop_last=True,num_workers=num_workers)
        model.train()
        loss_avg = []
        for _,batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if EMA:
                ema.update(model)



            loss_avg.append(loss.item())
            if not early_stop:
                scheduler.step()

        loss_avg = np.mean(loss_avg)  
        if not fulldata_train:  #validate with oof
            if EMA:
                f1,val_loss,logits,sequence_id = validate(ema.ema_model,val_loader,criterion,device)
            else:
                f1,val_loss,logits,sequence_id = validate(model,val_loader,criterion,device)
            if f1>best_f1:
                best_f1 = f1
                best_epoch = epoch
                if EMA:
                    best_model_weight = ema.ema_model.state_dict()
                else:
                    best_model_weight = model.state_dict()
                best_logits = logits
            
            elif early_stop and (((epoch-best_epoch)>patient) and (epoch>=50)):
                break

            if log:
                if early_stop:
                    lr_now = lr
                else:
                    lr_now = scheduler.get_last_lr()[0]

                elapsed_time = time.time() - start_time
                logging.info(f"Epoch {epoch+1}|{epoch_num},      lr:{lr_now:.2e},    time:{elapsed_time:.0f}      train/val_loss: {loss_avg:.4f}|{np.mean(val_loss):.4f},         f1:{f1:.4f},        best_f1:{best_f1:.4f}")


        else:
            elapsed_time = time.time() - start_time
            logging.info(f"Epoch {epoch+1}|{epoch_num}, train_loss: {loss_avg:.4f}")
    
    if not early_stop or fulldata_train:  #save the last step model
        if EMA:
            torch.save(ema.ema_model.state_dict(), f"{model_dir}model_weight_fold{tag}.pth")
        else:
            torch.save(model.state_dict(), f"{model_dir}model_weight_fold{tag}.pth")
    else:
        torch.save(best_model_weight, f"{model_dir}model_weight_fold{tag}.pth")



    if not fulldata_train:
        if early_stop:
            return best_f1,best_epoch,best_logits,sequence_id
        else:
            return f1,epoch,logits,sequence_id
    else:
        return