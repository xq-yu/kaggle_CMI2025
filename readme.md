Below you can find a outline of how to reproduce my solution for the [CMI - Detect Behavior with Sensor Data](https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/overview) competition.  

# ARCHIVE CONTENTS
|file|usage|
|---|---|
|source/main_imu.py|code to rebuild imu_only models|
|source/main_all.py|code to rebuild all_sensor models|
|source/config.py|params to be set|
|source/util/inference.py|code to generate predictions from trained model|


# HARDWARE
```
Operating System Information:
  Platform: Ubuntu 20.04.6 LTS
  Architecture: x86_64
  CPU Cores: 12
  Total Memory: 32 GB
  GPU: NVIDIA GeForce RTX 4090
  GPU Mem:24G
```

# SOFTWARE
```
GPU:
  CUDA Version: 12.6
  Driver Version: 560.94

Python Environment:
  Python Version: 3.10.16

Python packages are detailed separately in `requirements.txt`
```

# DATA SETUP (assumes the [Kaggle API](https://github.com/Kaggle/kaggle-api) is installed)
Below are the shell commands used in each step, as run from the top level directory  
The directory below should be consistent with config.py
```sh
# Prepare data. check if your data have been put in `/cmi-detect-behavior-with-sensor-data`(can be set in config.py)
kaggle competitions download -c cmi-detect-behavior-with-sensor-data
kaggle competitions download -c cmi-detect-behavior-with-sensor-data
kaggle competitions download -c cmi-detect-behavior-with-sensor-data

# project initialize
cd source
python project_init.py


```
# Train
1. The files in `model_dir`(default 'model/all_sensor/seed_x' and 'model/only_IMU/seed_x')set in config.py, could be overwrite.
**==Make sure the file in these directories have been backedup if you still need.==**
``` sh
cd source
python main_imu.py 1  # where 1 means the random seed
python main_all.py 1  # where 1 means the random seed
```
2. You can download the trained models [here](https://www.kaggle.com/datasets/minerppdy/cmi-minerppdy-model/)

# Inference
```python
import sys
import os
from util.inference import Classifier as MinerppdyClassifier
sys.path.append(os.path.join('prject_dir'))


################################
# load_model
################################

# at least one imu_only model should be selected
imu_model_dir = [
    '/model/only_IMU/seed_1/',
    '/model/only_IMU/seed_2/',
    '/model/only_IMU/seed_3/'
]
# at least one all_sensor model should be selected
all_model_dir = [
    '/model/all_sensor/seed_1/',
    '/model/all_sensor/seed_2/',
    '/model/all_sensor/seed_3/'
]
model_dir = imu_model_dir + all_model_dir
minerppdy_cls = MinerppdyClassifier(
    model_dir,
    device,
    seq_len=128,
    imu_weight=0.3,
    all_weight=0.7
)

print('imu_model_num', len(minerppdy_cls.imu_only_models))
print('all_model_num', len(minerppdy_cls.all_sensor_models))

################################
# prediction function 
################################
minerppdy_cls.predict(sequence.copy(), demographics.copy())
```