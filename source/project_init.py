import pandas as pd
import numpy as np
from config import data_dir,gesture_list



def create_sample(data_dir,gesture_list):
    df = pd.read_csv(data_dir+'/train.csv')
    demo = pd.read_csv(data_dir+'/train_demographics.csv')

    sample = pd.merge(df,demo,how='left',on='subject')[['sequence_id','subject','handedness','gesture']].drop_duplicates()

    gesture_map = {k:i for i,k in enumerate(gesture_list)}
    sample['y'] = sample.gesture.map(gesture_map)
    sample.to_csv(data_dir+'/sample.csv',index=False)
    

if __name__=='__main__':
    create_sample(data_dir,gesture_list)