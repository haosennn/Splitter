## Joint Optimization of Splitter Pattern and Image Reconstruction for Metasurface-based Color Imaging Systems

### Setup and Installation
```
# create and activate new conda environment
conda create -n splitter python=3.10
conda activate splitter

# install pytorch (version: 2.1.1)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# install required packages
pip install -r requirements.txt

# clone the repository
git clone https://github.com/haosennn/Splitter.git
cd ./Splitter/
```

### Dataset
Training and test datasets can be downloaded from the following websites:
1. DIV2K: https://data.vision.ee.ethz.ch/cvl/DIV2K/
2. Kodak24: https://r0k.us/graphics/kodak/
3. McMaster: https://www4.comp.polyu.edu.hk/~cslzhang/CDM_Dataset.htm

### Pre-trained Model
Pretrained models can be downloaded from: https://drive.google.com/drive/folders/1F_cTzA7KaeKz6eEpiT-CEuQozV7WtRL9?usp=drive_link

### Validate Pre-trained Models
python main_test.py --exp 1 --save --sigmas 0 --data_test Kodak24

### Train Joint-Network
python main_train.py --S_type A --gpu_ids 0
