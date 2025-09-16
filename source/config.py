data_dir = '../cmi-detect-behavior-with-sensor-data/'
folds_info_file = f'./team_folds.csv'

gesture_list = [
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


class Config_all:
    fold_num = 5
    training_seed = 1
    imu_only = False

    data_dir = data_dir
    folds_info_file = folds_info_file
    output_dir = f'../model/all_sensor/'
    fea_config = {'imu_dim':34,
                 'thmtof_dim':5+64*5}
    loss_config = {
        'aux_weight':0.2
    }
    
    aug_config = {
        'if_aug':False,
        'imu_rotate_prop':0.6,
        'stretch_prop':0.5,
        'shift_prop':0.5,
        'rot_drop_prop':0.05,
        'tof_drop_prop':0.5
    }

    train_config = {
        'imu_only':imu_only,
        'num_class':18,
        'num_workers':0,
        'model_dir':output_dir+f'seed_{training_seed}/',
        'batch_size':128,
        'early_stop':False,
        'patient':10,
        'epoch_num':50,
        'lr':0.001,
        'weight_decay':1e-4,
        'warmup_prop':0.001,
        'EMA':False,
        'ema_decay':0.99,
    }

    gesture_list = gesture_list




class Config_imu:
    fold_num = 5
    training_seed = 1
    imu_only = True

    data_dir = data_dir
    folds_info_file = folds_info_file
    output_dir = f'../model/only_IMU/'
    
    fea_config = {'imu_dim':34,
                 'thmtof_dim':5+64*5}
    
    loss_config = {
        'aux_weight':0.2
    }
    
    aug_config = {
        'if_aug':True,
        'imu_rotate_prop':0.7,
        'stretch_prop':0.6,
        'shift_prop':0.6,
        'rot_drop_prop':0,
        'tof_drop_prop':0.2
    }

    train_config = {
        'imu_only':imu_only,
        'num_class':18,
        'num_workers':0,
        'model_dir':output_dir+f'seed_{training_seed}/',
        'batch_size':512,
        'early_stop':False,
        'patient':10,
        'epoch_num':100,
        'lr':0.001,
        'weight_decay':1e-4,
        'warmup_prop':0.001,

        'EMA':True,
        'ema_decay':0.99,

    }

    gesture_list = gesture_list
