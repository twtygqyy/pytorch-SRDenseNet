# PyTorch SRDenseNet
Implementation of ICCV 2017 Paper: [Image Super-Resolution Using Dense Skip Connections](http://openaccess.thecvf.com/content_iccv_2017/html/Tong_Image_Super-Resolution_Using_ICCV_2017_paper.html) in PyTorch

## Usage
### Training
```
usage: main.py [-h] [--batchSize BATCHSIZE] [--nEpochs NEPOCHS] [--lr LR]
               [--step STEP] [--cuda] [--resume RESUME]
               [--start-epoch START_EPOCH] [--threads THREADS]
               [--momentum MOMENTUM] [--weight-decay WEIGHT_DECAY]
               [--pretrained PRETRAINED]

PyTorch DenseNet

optional arguments:
  -h, --help            show this help message and exit
  --batchSize BATCHSIZE
                        training batch size
  --nEpochs NEPOCHS     number of epochs to train for
  --lr LR               Learning Rate. Default=1e-4
  --step STEP           Sets the learning rate to the initial LR decayed by
                        momentum every n epochs, Default: n=10
  --cuda                Use cuda?
  --resume RESUME       Path to checkpoint (default: none)
  --start-epoch START_EPOCH
                        Manual epoch number (useful on restarts)
  --threads THREADS     Number of threads for data loader to use, Default: 1
  --momentum MOMENTUM   Momentum, Default: 0.9
  --weight-decay WEIGHT_DECAY, --wd WEIGHT_DECAY
                        weight decay, Default: 1e-4
  --pretrained PRETRAINED
                        path to pretrained model (default: none)
```
An example of training usage is shown as follows:
```
python main.py --cuda
```

### Prepare Training dataset
  - Please refer [Code for Data Generation](https://github.com/twtygqyy/pytorch-SRResNet/tree/master/data) for creating training files.
  - Data augmentations including flipping, rotation, downsizing are adopted.

### Performance
  - So far performance in PSNR is not as good as paper, since we only used 30,000 images for training while the authors used 50,000 images.
  
| Dataset        | SRDenseNet Paper          | SRDenseNet PyTorch|
| ------------- |:-------------:| -----:|
| Set5      | 32.02      | **31.58** |
| Set14     | 28.50      | **28.36** |
| BSD100    | 27.53      | **27.38** |

### Misc.
  - L1 Charbonnier loss from [Deep Laplacian Pyramid Networks for Fast and Accurate Super-Resolution](https://arxiv.org/abs/1704.03915) is applied instead of MSE loss, and PReLu is applied instead of ReLu due to gradient vanishing.
