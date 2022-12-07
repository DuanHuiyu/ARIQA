## CFIQA & ARIQA Models

## (0) Dependencies/Setup

### Installation

- Clone this repo:
```bash
git clone https://github.com/DuanHuiyu/ARIQA
cd ARIQA/code3_cfiqa_ariqa
```

- Install PyTorch 1.0+ and torchvision fom http://pytorch.org
```bash
conda create -n ARIQA pip
conda activate ARIQA
pip install -r requirements.txt
```

## (1) Training and Evaluation for CFIQA

### Spliting the databases into training and evaluation sets
```
python generate_cfiqa_csv.py
```

### deep baseline models
```
python baselines_cfiqa.py --net 'squeeze' --name baseline --use_gpu --data_dir '../database1_cfiqa'
python baselines_cfiqa.py --net 'alex' --name baseline --use_gpu --data_dir '../database1_cfiqa'

python baselines_cfiqa.py --net 'vgg16' --name baseline --use_gpu --data_dir '../database1_cfiqa'
python baselines_cfiqa.py --net 'vgg19' --name baseline --use_gpu --data_dir '../database1_cfiqa'
python baselines_cfiqa.py --net 'vgg16_plus' --name baseline --use_gpu --data_dir '../database1_cfiqa'
# below two lines are used to find best layers
python baselines_cfiqa.py --net 'vgg19_all_layers' --name baseline --use_gpu --data_dir '../database1_cfiqa'
python baselines_cfiqa.py --net 'vgg16_all_layers' --name baseline --use_gpu --data_dir '../database1_cfiqa'

python baselines_cfiqa.py --net 'resnet18' --name baseline --use_gpu --data_dir '../database1_cfiqa'
python baselines_cfiqa.py --net 'resnet34' --name baseline --use_gpu --data_dir '../database1_cfiqa'
python baselines_cfiqa.py --net 'resnet50' --name baseline --use_gpu --data_dir '../database1_cfiqa'
```

### deep baseline models (baseline from zhang et al. (LPIPS), i.e., using LPIPS trained weights)
```
python baselines_cfiqa_lpips.py --net 'squeeze' --name baseline_lpips --use_gpu --data_dir '../database1_cfiqa'
python baselines_cfiqa_lpips.py --net 'alex' --name baseline_lpips --use_gpu --data_dir '../database1_cfiqa'
python baselines_cfiqa_lpips.py --net 'vgg' --name baseline_lpips --use_gpu --data_dir '../database1_cfiqa'
```

### CFIQA model
```
python train_cfiqa.py --use_gpu --net 'squeeze' --name cfiqa_squeeze --batch_size 10 --nepoch 100 --nepoch_decay 50 --save_epoch_freq 20 --data_dir '../database1_cfiqa' --cross_num 2
python train_cfiqa.py --use_gpu --net 'alex' --name cfiqa_alex --batch_size 10 --nepoch 100 --nepoch_decay 50 --save_epoch_freq 20 --data_dir '../database1_cfiqa' --cross_num 2
python train_cfiqa.py --use_gpu --net 'vgg' --name cfiqa_vgg --batch_size 10 --nepoch 100 --nepoch_decay 50 --save_epoch_freq 20 --data_dir '../database1_cfiqa' --cross_num 2
python train_cfiqa.py --use_gpu --net 'vgg19' --name cfiqa_vgg19 --batch_size 10 --nepoch 100 --nepoch_decay 50 --save_epoch_freq 20 --data_dir '../database1_cfiqa' --cross_num 2
python train_cfiqa.py --use_gpu --net 'resnet18' --name cfiqa_resnet18 --batch_size 10 --nepoch 100 --nepoch_decay 50 --save_epoch_freq 20 --data_dir '../database1_cfiqa' --cross_num 2
python train_cfiqa.py --use_gpu --net 'resnet34' --name cfiqa_resnet34 --batch_size 10 --nepoch 100 --nepoch_decay 50 --save_epoch_freq 20 --data_dir '../database1_cfiqa' --cross_num 2
python train_cfiqa.py --use_gpu --net 'resnet50' --name cfiqa_resnet50 --batch_size 10 --nepoch 100 --nepoch_decay 50 --save_epoch_freq 20 --data_dir '../database1_cfiqa' --cross_num 2
```

### CFIQA plus model
```
python train_cfiqa_plus.py --use_gpu --net 'resnet34' --name cfiqa_plus_resnet34 --batch_size 10 --nepoch 100 --nepoch_decay 50 --save_epoch_freq 20 --data_dir '../database1_cfiqa' --cross_num 2
```

## (2) Training and Evaluation for ARIQA

### Spliting the databases into training and evaluation sets
```
python generate_ariqa_csv.py
```

### deep baseline models
```
python baselines_ariqa.py --net 'squeeze' --name baseline --use_gpu --data_dir '../database2_ariqa'
python baselines_ariqa.py --net 'alex' --name baseline --use_gpu --data_dir '../database2_ariqa'

python baselines_ariqa.py --net 'vgg16' --name baseline --use_gpu --data_dir '../database2_ariqa'
python baselines_ariqa.py --net 'vgg19' --name baseline --use_gpu --data_dir '../database2_ariqa'
python baselines_ariqa.py --net 'vgg16_plus' --name baseline --use_gpu --data_dir '../database2_ariqa'

python baselines_ariqa.py --net 'resnet18' --name baseline --use_gpu --data_dir '../database2_ariqa'
python baselines_ariqa.py --net 'resnet34' --name baseline --use_gpu --data_dir '../database2_ariqa'
python baselines_ariqa.py --net 'resnet50' --name baseline --use_gpu --data_dir '../database2_ariqa'
```

### deep baseline models (baseline from zhang et al. (LPIPS), i.e., using LPIPS trained weights)
```
python baselines_ariqa_lpips.py --net 'squeeze' --name baseline_lpips --use_gpu --data_dir '../database2_ariqa'
python baselines_ariqa_lpips.py --net 'alex' --name baseline_lpips --use_gpu --data_dir '../database2_ariqa'
python baselines_ariqa_lpips.py --net 'vgg' --name baseline_lpips --use_gpu --data_dir '../database2_ariqa'
```

### ARIQA model
```
python train_ariqa.py --use_gpu --net 'resnet34' --name ariqa --batch_size 2 --nepoch 30 --nepoch_decay 10 --save_epoch_freq 5 --data_dir '../database2_ariqa' --cross_num 0
python train_ariqa.py --use_gpu --net 'resnet34' --name ariqa --batch_size 2 --nepoch 30 --nepoch_decay 10 --save_epoch_freq 5 --data_dir '../database2_ariqa' --cross_num 1
python train_ariqa.py --use_gpu --net 'resnet34' --name ariqa --batch_size 2 --nepoch 30 --nepoch_decay 10 --save_epoch_freq 5 --data_dir '../database2_ariqa' --cross_num 2
python train_ariqa.py --use_gpu --net 'resnet34' --name ariqa --batch_size 2 --nepoch 30 --nepoch_decay 10 --save_epoch_freq 5 --data_dir '../database2_ariqa' --cross_num 3
python train_ariqa.py --use_gpu --net 'resnet34' --name ariqa --batch_size 2 --nepoch 30 --nepoch_decay 10 --save_epoch_freq 5 --data_dir '../database2_ariqa' --cross_num 4
```

### ARIQA plus model
```
python train_ariqa_plus.py --use_gpu --net 'resnet34' --name ariqa_plus --batch_size 2 --nepoch 30 --nepoch_decay 10 --save_epoch_freq 5 --data_dir '../database2_ariqa' --cross_num 0
python train_ariqa_plus.py --use_gpu --net 'resnet34' --name ariqa_plus --batch_size 2 --nepoch 30 --nepoch_decay 10 --save_epoch_freq 5 --data_dir '../database2_ariqa' --cross_num 1
python train_ariqa_plus.py --use_gpu --net 'resnet34' --name ariqa_plus --batch_size 2 --nepoch 30 --nepoch_decay 10 --save_epoch_freq 5 --data_dir '../database2_ariqa' --cross_num 2
python train_ariqa_plus.py --use_gpu --net 'resnet34' --name ariqa_plus --batch_size 2 --nepoch 30 --nepoch_decay 10 --save_epoch_freq 5 --data_dir '../database2_ariqa' --cross_num 3
python train_ariqa_plus.py --use_gpu --net 'resnet34' --name ariqa_plus --batch_size 2 --nepoch 30 --nepoch_decay 10 --save_epoch_freq 5 --data_dir '../database2_ariqa' --cross_num 4
```

## Acknowledgements

This repository borrows heavily from the [LPIPS](https://github.com/richzhang/PerceptualSimilarity) repository.