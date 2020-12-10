# Viet Nam celebrity recognition system
This documents describe steps required for building and test the system: 

1. [Set up environment](#set-up-environment)
2. [Install basic packages and AI packages](#install-basic-packages-and-ai-packages)
3. [Download pre-trained AI models for recognition process](#download-trained-ai-models-for-recognition-process)
4. [Run script tests to check whether the system work or not](#run-script-tests-to-check-whether-the-system-work-or-not)

## Set up environment
The following example shows the use of `venv` in python pip: 
```
python -m venv vn_celeb_env
source vn_celeb_env/bin/activate
```
**Please make sure that the python version >= 3.6**
## Install basic packages and AI packages
Clone the repository and enter to root folder of it:
```
git clone <gitlab_link>
cd VN_celeb_face_vs_emotion
```
Install basic packages:
```
pip install -r requirements.txt
```
Install AI packages:
```
pip install --find-links=https://download.pytorch.org/whl/torch_stable.html -r requirements_ai_lib.txt
```
Please note that these AI packages are **only compatible with CUDA 11.1**

## Download trained AI models for recognition process
In root directory of project create weights folder :
```
mkdir weights weights/detection weights/encoder weights/emotion weights/mlp
```
Install `gdown` for downloading models file from **Google Drive**
```
pip install gdown
```
Download trained models for tasks: 
1. MLP models:
```
gdown https://drive.google.com/uc?id=1CfK8whYAhelw6O3-SStN59-_f1tNX_N7 -O weights/mlps/mlp_models.zip
unzip -q weights/mlp/mlp_models.zip -d weights/mlp/
rm weights/mlp/mlp_models.zip
```
2. Detection models:
```
gdown https://drive.google.com/uc?id=1dH34cqbcLgFnnFovUcCRAhwwP1ANfpdi -O weights/detection/detection_models.zip 
unzip -q weights/detection/detection_models.zip -d weights/detection 
rm weights/detection/detection_models.zip
```
3. Encoder model
```
gdown https://drive.google.com/uc\?id\=1-SB7kAJj1e-4SN6hiFBP1rQN54EVQRf0 -O weights/encoder/encoder_52_celeb.pth
```
4. Emotion model
```
gdown https://drive.google.com/uc?id=1bx4rIH4KKLrgZnZ2RpXaIvDs9tzjHkYf -O weights/emotion/emotion_690_cls.pth 
```

## Run script tests to check whether the system work or not
In root dir of project create `scripts` folder:
```
mkdir scripts
```
Download test script from **Google Drive**, place it to this `scripts` folder and grant executable permissions to the file:
```
gdown https://drive.google.com/uc?id=1Sa7Y8K4sJF-fV6w4Euds70bv0F8l_-aW -O scripts/ensembles_stat.txt
chmod 777 scripts/ensembles_stat.txt
```
Run `ensembles_stat.txt` file:
```
./scripts/ensembles_stat.txt
```
Check folder `output_demo` and enter to `run_id` (example: `1210_120012`) folder to get results file:
1. `info.txt`: log file when running test script
2. `tracker_rap_viet_intro_sg_gpu_1.csv`: csv file for doing statistic
3. `tracker_rap_viet_intro_sg_gpu_1.json`: JSON indexing file, this is the **output** file of indexing process.
