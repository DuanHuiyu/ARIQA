## ------------------------------------------------------------------------------------------
## Confusing image quality assessment: Towards better augmented reality experience
## Huiyu Duan, Xiongkuo Min, Yucheng Zhu, Guangtao Zhai, Xiaokang Yang, and Patrick Le Callet
## IEEE Transactions on Image Processing (TIP)
## ------------------------------------------------------------------------------------------

import torch.backends.cudnn as cudnn
cudnn.benchmark=False

import numpy as np
import time
import os
import argparse
from IPython import embed

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
import torch.backends.cudnn as cudnn
import torch.nn.functional as F 
import torch.utils.model_zoo as model_zoo

import dataset_cfiqa_ariqa

from calc_performance import IQAPerformance

import pandas as pd

from lpips import trainer_ariqa_sal as trainer

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='lpips', help='distance model type [lpips] for linearly calibrated net, [baseline] for off-the-shelf network, [l2] for euclidean distance, [ssim] for Structured Similarity Image Metric')
parser.add_argument('--net', type=str, default='alex', help='[squeeze], [alex], or [vgg] for network architectures')
parser.add_argument('--batch_size', type=int, default=50, help='batch size to test image patches in')
parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')
parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0], help='gpus to use')

parser.add_argument('--nepoch', type=int, default=5, help='# epochs at base learning rate')
parser.add_argument('--nepoch_decay', type=int, default=5, help='# additional epochs at linearly learning rate')
parser.add_argument('--display_freq', type=int, default=5000, help='frequency (in instances) of showing training results on screen')
parser.add_argument('--print_freq', type=int, default=5000, help='frequency (in instances) of showing training results on console')
parser.add_argument('--save_latest_freq', type=int, default=20000, help='frequency (in instances) of saving the latest results')
parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')
parser.add_argument('--display_id', type=int, default=0, help='window id of the visdom display, [0] for no displaying')
parser.add_argument('--display_winsize', type=int, default=256, help='display window size')
parser.add_argument('--display_port', type=int, default=8001,  help='visdom display port')
parser.add_argument('--use_html', action='store_true', help='save off html pages')
parser.add_argument('--checkpoints_dir', type=str, default='results/checkpoints', help='checkpoints directory')
parser.add_argument('--name', type=str, default='tmp', help='directory name for training')

parser.add_argument('--from_scratch', action='store_true', help='model was initialized from scratch')
parser.add_argument('--train_trunk', action='store_true', help='model trunk was trained/tuned')
parser.add_argument('--train_plot', action='store_true', help='plot saving')

parser.add_argument('--data_dir', dest='data_dir', help='Directory path for testing reference image data.',
        default='../database2_ariqa', type=str)
parser.add_argument('--train_filename_list', dest='train_filename_list', help='Path to text file containing relative paths for every example.',
        default='./train_test_split/ARIQA/ARIQA_train', type=str)
parser.add_argument('--test_filename_list', dest='test_filename_list', help='Path to text file containing relative paths for every example.',
        default='./train_test_split/ARIQA/ARIQA_test', type=str)
parser.add_argument('--cross_num', dest='cross_num', help='cross number.',
        default=0, type=int)

opt = parser.parse_args()
opt.save_dir = os.path.join(opt.checkpoints_dir,opt.name)
if(not os.path.exists(opt.save_dir)):
    os.makedirs(opt.save_dir)

# initialize model
trainer = trainer.Trainer()
print(opt.net)

# load data from all training sets
cross = opt.cross_num
test_results = []
mos_list = []


trainer.initialize(model=opt.model, net=opt.net, use_gpu=opt.use_gpu, is_train=True, 
    pnet_rand=opt.from_scratch, pnet_tune=opt.train_trunk, gpu_ids=opt.gpu_ids)

img_size = 336
transformations = transforms.Compose([transforms.Resize(img_size),transforms.ToTensor(),transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
train_filename_list = opt.train_filename_list+'_'+str(cross)+'.csv'
ARIQA_dataset_train = dataset_cfiqa_ariqa.FRIQA_ARBG_Dataset(opt.data_dir, train_filename_list, transformations, size=img_size, is_train=True, saliency=True)
train_loader = torch.utils.data.DataLoader(dataset=ARIQA_dataset_train,batch_size=opt.batch_size,shuffle=True,num_workers=6)

transformations = transforms.Compose([transforms.Resize(img_size),transforms.ToTensor(),transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
test_filename_list = opt.test_filename_list+'_'+str(cross)+'.csv'
ARIQA_dataset_test = dataset_cfiqa_ariqa.FRIQA_ARBG_Dataset_test(opt.data_dir,test_filename_list,transformations, size=img_size, is_train=False, saliency=True)
test_loader = torch.utils.data.DataLoader(dataset=ARIQA_dataset_test,batch_size=1,shuffle=False,num_workers=0)

dataset_size = len(ARIQA_dataset_train)


print('---------')
print('starting!')
print('---------')
print(cross)

temp_srocc = 0
total_steps = 0
for epoch in range(1, opt.nepoch + opt.nepoch_decay + 1):
    print('cross:'+str(cross))
    print('epoch:'+str(epoch))
    epoch_start_time = time.time()
    for i, data in enumerate(train_loader):
        if i%20 == 0:
            print('iteration:'+str(i))
        iter_start_time = time.time()
        total_steps += opt.batch_size
        epoch_iter = total_steps - dataset_size * (epoch - 1)

        trainer.set_input(data)
        trainer.optimize_parameters()

    trainer.save(opt.save_dir, 'cross'+str(cross)+'_latest')

    print('End of epoch %d / %d \t Time Taken: %d sec' %
        (epoch, opt.nepoch + opt.nepoch_decay, time.time() - epoch_start_time))

    if epoch > opt.nepoch:
        trainer.update_learning_rate(opt.nepoch_decay)


    # testing
    total_loss = 0
    test_mos_predict = []
    test_mos_origin = []

    # switch to evaluate mode
    # model.eval()

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            # calculate predicted mos on testing set
            trainer.set_test_input(data)
            loss,mos = trainer.test()

            mos_predict = loss

            # append predicted mos and mos
            test_mos_predict.append(np.reshape(mos_predict.cpu().numpy(),(1)))
            test_mos_origin.append(np.reshape(mos.cpu().numpy(),(1)))
    
    iqa = IQAPerformance(test_mos_origin,test_mos_predict)
    print(iqa.compute())

    iqa_result = iqa.compute()
    temp_result = []
    data_frame = []
    temp_result.append(cross+1)
    temp_result.append(epoch)
    temp_result.append(iqa_result[0])
    temp_result.append(iqa_result[1])
    temp_result.append(iqa_result[2])
    temp_result.append(iqa_result[3])
    temp_result.append(iqa_result[4])
    temp_result.append(iqa_result[5])
    temp_result.append(iqa_result[6])
    temp_result.append(iqa_result[7])

    column_dataframe = ['cross','epoch','srocc','krocc','plcc','rmse','mae','srocc','krocc','plcc']
    data_frame.append(temp_result)
    df = pd.DataFrame(data_frame,columns = column_dataframe)
    df.to_csv(os.path.join(opt.save_dir,'cross_result'+str(cross)+'.csv'),mode='a+',header=False,index=False)

    if temp_srocc < iqa_result[5]:  # scrocc
        temp_srocc = iqa_result[5]
        print('saving the b-e-s-t model at the end of epoch %d, iters %d' %
            (epoch, total_steps))
        trainer.save(opt.save_dir, 'cross'+str(cross)+'_best')
        test_mos_predict_best = test_mos_predict

        test_dataframe = []
        for result in test_mos_predict_best:
            result = result[0]
            each_row = []
            each_row.append(result)
            test_dataframe.append(each_row)
        df = pd.DataFrame(test_dataframe,columns = ['results'])
        df.to_csv(os.path.join(opt.save_dir,'test'+str(cross)+'_best.csv'),header=False,index=False)


test_results = test_results+test_mos_predict_best
mos_list = mos_list+test_mos_origin


iqa = IQAPerformance(mos_list,test_results)
print(iqa.compute())

np.save(os.path.join(opt.save_dir,'test'+str(cross)+'.npy'), test_results)
test_dataframe = []
for result in test_results:
    result = result[0]
    each_row = []
    each_row.append(result)
    test_dataframe.append(each_row)
df = pd.DataFrame(test_dataframe,columns = ['results'])
df.to_csv(os.path.join(opt.save_dir,'test'+str(cross)+'.csv'),header=False,index=False)