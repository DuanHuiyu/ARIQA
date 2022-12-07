## ------------------------------------------------------------------------------------------
## Confusing image quality assessment: Towards better augmented reality experience
## Huiyu Duan, Xiongkuo Min, Yucheng Zhu, Guangtao Zhai, Xiaokang Yang, and Patrick Le Callet
## IEEE Transactions on Image Processing (TIP)
## ------------------------------------------------------------------------------------------

import os
import numpy as np
import cv2
import pandas as pd

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms

from util.utils import DataAugmentation


def get_list_from_filenames(file_path):

    with open(file_path) as f:
        lines = f.read().splitlines()

    return lines

class FRIQA_Dataset(Dataset):
    def __init__(self, data_dir, csv_path, transform, suff='.png', is_train=True, saliency=False):
        column_names = ['DisImg','RefImg','MOS']
        tmp_df = pd.read_csv(csv_path, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")
        self.data_dir = data_dir
        self.transform = transform
        self.img_size = 512
        self.X_train_dis = tmp_df[['DisImg']]
        self.X_train_ref = tmp_df[['RefImg']]
        self.Y_train = tmp_df['MOS']

        self.length = len(tmp_df)

        self.saliency = saliency

        if is_train:
            self.augm = DataAugmentation(
                img_size=self.img_size,
                with_random_hflip=True,
                with_random_crop=True)
        else:
            self.augm = DataAugmentation(
                img_size=self.img_size,
                with_random_hflip=False,
                with_random_crop=False)

        # self.data_dir_dis = data_dir_dis
        # self.data_dir_ref = data_dir_ref
        # # self.dis = glob.glob(os.path.join(self.data_dir_dis, '*'+suff))
        # # self.ref = glob.glob(os.path.join(self.data_dir_ref, '*'+suff))
        # self.dis = []
        # self.ref = []
        # for i in range(300):
        #     self.dis.append(os.path.join(self.data_dir_dis,'{:06d}.png'.format(i+1)))
        #     self.ref.append(os.path.join(self.data_dir_ref,'{:06d}.png'.format(i+1)))

    def __getitem__(self, index):
        # this_dir = os.path.join(self.data_dir,self.X_train_dis.iloc[index,0])
        this_dir = self.data_dir+self.X_train_dis.iloc[index,0]
        # img_dis = Image.open(this_dir)
        # img_dis = img_dis.convert('RGB')
        img_dis = cv2.imread(this_dir, cv2.IMREAD_COLOR)
        img_dis = cv2.cvtColor(img_dis, cv2.COLOR_BGR2RGB)

        # this_dir = self.ref[index]
        # this_dir = os.path.join(self.data_dir,self.X_train_ref.iloc[index,0])
        this_dir = self.data_dir+self.X_train_ref.iloc[index,0]
        # img_ref = Image.open(this_dir)
        # img_ref = img_ref.convert('RGB')
        img_ref = cv2.imread(this_dir, cv2.IMREAD_COLOR)
        img_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2RGB)

        if self.saliency:
            saliency_map_dir = this_dir.replace('/A', '/A_saliency')
            saliency_map_dir = saliency_map_dir.replace('/B', '/B_saliency')    # it mean replace "A" or "B" to "A_saliency" or "B_saliency", respectively
            saliency_map = cv2.imread(saliency_map_dir, cv2.IMREAD_COLOR)
            saliency_map = cv2.cvtColor(saliency_map, cv2.COLOR_BGR2GRAY)
            
        if self.saliency:
            img_dis,img_ref,saliency_map = self.augm.transform_triplets(img_dis, img_ref, saliency_map)
        else:
            img_dis,img_ref = self.augm.transform_twins(img_dis, img_ref)

        if self.transform is not None:
            img_dis = self.transform(img_dis)
            img_ref = self.transform(img_ref)
            if self.saliency:
                transformations = transforms.Compose([transforms.Resize(224),transforms.ToTensor()])
                saliency_map = transformations(saliency_map)

        
        y_mos = self.Y_train.iloc[index]
        y_label = torch.FloatTensor(np.array(float(y_mos)))

        if self.saliency:
            return img_dis,img_ref,saliency_map,y_label
        else:
            return img_dis,img_ref,y_label

    def __len__(self):
        return self.length



class FRIQA_twoafc_Dataset(Dataset):
    def __init__(self, data_dir, csv_path1, csv_path2, transform, suff='.png', is_train=True, saliency=False):
        column_names = ['DisImg','RefImg','MOS']
        tmp_df1 = pd.read_csv(csv_path1, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")
        tmp_df2 = pd.read_csv(csv_path2, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")
        self.data_dir = data_dir
        self.transform = transform
        self.img_size = 512
        self.X_train_dis = tmp_df1[['DisImg']]
        self.X_train_ref1 = tmp_df1[['RefImg']]
        self.X_train_ref2 = tmp_df2[['RefImg']]
        self.Y_train1 = tmp_df1['MOS']
        self.Y_train2 = tmp_df2['MOS']

        self.saliency = saliency

        self.length = len(tmp_df1)

        if is_train:
            self.augm = DataAugmentation(
                img_size=self.img_size,
                with_random_hflip=True,
                with_random_crop=True)
        else:
            self.augm = DataAugmentation(
                img_size=self.img_size,
                with_random_hflip=False,
                with_random_crop=False)


    def __getitem__(self, index):
        # this_dir = os.path.join(self.data_dir,self.X_train_dis.iloc[index,0])
        img_dis_dir = self.data_dir+self.X_train_dis.iloc[index,0]
        img_dis = cv2.imread(img_dis_dir, cv2.IMREAD_COLOR)
        img_dis = cv2.cvtColor(img_dis, cv2.COLOR_BGR2RGB)

        img_ref1_dir = self.data_dir+self.X_train_ref1.iloc[index,0]
        img_ref1 = cv2.imread(img_ref1_dir, cv2.IMREAD_COLOR)
        img_ref1 = cv2.cvtColor(img_ref1, cv2.COLOR_BGR2RGB)

        img_ref2_dir = self.data_dir+self.X_train_ref2.iloc[index,0]
        img_ref2 = cv2.imread(img_ref2_dir, cv2.IMREAD_COLOR)
        img_ref2 = cv2.cvtColor(img_ref2, cv2.COLOR_BGR2RGB)

        
        if self.saliency:
            saliency_map1_dir = img_ref1_dir.replace('/A', '/A_saliency')
            saliency_map1 = cv2.imread(saliency_map1_dir, cv2.IMREAD_COLOR)
            saliency_map1 = cv2.cvtColor(saliency_map1, cv2.COLOR_BGR2GRAY)
            saliency_map2_dir = img_ref2_dir.replace('/B', '/B_saliency')
            saliency_map2 = cv2.imread(saliency_map2_dir, cv2.IMREAD_COLOR)
            saliency_map2 = cv2.cvtColor(saliency_map2, cv2.COLOR_BGR2GRAY)
            

        if self.saliency:
            img_dis,img_ref1,img_ref2,saliency_map1,saliency_map2 = self.augm.transform_quintuplets(img_dis,img_ref1,img_ref2,saliency_map1,saliency_map2)
        else:
            img_dis,img_ref1,img_ref2 = self.augm.transform_triplets(img_dis,img_ref1,img_ref2)

        if self.transform is not None:
            img_dis = self.transform(img_dis)
            img_ref1 = self.transform(img_ref1)
            img_ref2 = self.transform(img_ref2)
            if self.saliency:
                transformations = transforms.Compose([transforms.Resize(224),transforms.ToTensor()])
                saliency_map1 = transformations(saliency_map1)
                # saliency_map1 = (self.transform(saliency_map1)+1.)/2.
                saliency_map2 = transformations(saliency_map2)
                # saliency_map2 = (self.transform(saliency_map2)+1.)/2.

        y_mos1 = self.Y_train1.iloc[index]
        y_label1 = torch.FloatTensor(np.array(float(y_mos1/10)))    # transform to [0,1]
        y_mos2 = self.Y_train2.iloc[index]
        y_label2 = torch.FloatTensor(np.array(float(y_mos2/10)))    # transform to [0,1]

        judge_img = (y_label1-y_label2+1.)/2. # transform to [0,1]


        # return img_dis,img_ref,y_label
        if self.saliency:
            return {'p0': img_ref1, 'p1': img_ref2, 'ref': img_dis, 'judge': judge_img, 'y0': y_label1, 'y1': y_label2, 's1': saliency_map1, 's2': saliency_map2,
                'p0_path': img_ref1_dir, 'p1_path': img_ref2_dir, 'ref_path': img_dis_dir}
        else:
            return {'p0': img_ref1, 'p1': img_ref2, 'ref': img_dis, 'judge': judge_img, 'y0': y_label1, 'y1': y_label2,
                'p0_path': img_ref1_dir, 'p1_path': img_ref2_dir, 'ref_path': img_dis_dir}

    def __len__(self):
        return self.length

class FRIQA_AR_Dataset(Dataset):
    def __init__(self, data_dir, csv_path, transform, size=224, is_train=True, saliency=False):
        column_names = ['RefImg','DisImg1','DisImg2','MOS1','MOS2']
        tmp_df = pd.read_csv(csv_path, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")
        self.data_dir = data_dir
        self.transform = transform
        self.img_size = size
        self.X_train_ref = tmp_df[['RefImg']]
        self.X_train_dis1 = tmp_df[['DisImg1']]
        self.X_train_dis2 = tmp_df[['DisImg2']]
        self.Y_train1 = tmp_df['MOS1']
        self.Y_train2 = tmp_df['MOS2']

        self.saliency = saliency

        self.length = len(tmp_df)

        if is_train:
            self.augm = DataAugmentation(
                img_size=self.img_size,
                with_random_hflip=True,
                with_random_crop=True,
                crop_scale = (0.8, 1.0),
                crop_ratio=(0.9, 1.1))
        else:
            self.augm = DataAugmentation(
                img_size=self.img_size,
                with_random_hflip=False,
                with_random_crop=False)


    def __getitem__(self, index):
        # this_dir = os.path.join(self.data_dir,self.X_train_dis.iloc[index,0])
        img_ref_dir = self.data_dir+self.X_train_ref.iloc[index,0]
        img_ref = cv2.imread(img_ref_dir, cv2.IMREAD_COLOR)
        img_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2RGB)

        img_dis1_dir = self.data_dir+self.X_train_dis1.iloc[index,0]
        img_dis1 = cv2.imread(img_dis1_dir, cv2.IMREAD_COLOR)
        img_dis1 = cv2.cvtColor(img_dis1, cv2.COLOR_BGR2RGB)

        img_dis2_dir = self.data_dir+self.X_train_dis2.iloc[index,0]
        img_dis2 = cv2.imread(img_dis2_dir, cv2.IMREAD_COLOR)
        img_dis2 = cv2.cvtColor(img_dis2, cv2.COLOR_BGR2RGB)

        
        if self.saliency:
            saliency_map1_dir = img_ref_dir.replace('/img_AR/reference', '/img_AR_sal')
            saliency_map1 = cv2.imread(saliency_map1_dir, cv2.IMREAD_COLOR)
            saliency_map1 = cv2.cvtColor(saliency_map1, cv2.COLOR_BGR2GRAY)
            saliency_map2_dir = img_ref_dir.replace('/img_AR/reference', '/img_AR_sal')
            saliency_map2 = cv2.imread(saliency_map2_dir, cv2.IMREAD_COLOR)
            saliency_map2 = cv2.cvtColor(saliency_map2, cv2.COLOR_BGR2GRAY)
            

        if self.saliency:
            img_ref,img_dis1,img_dis2,saliency_map1,saliency_map2 = self.augm.transform_quintuplets(img_ref,img_dis1,img_dis2,saliency_map1,saliency_map2)
        else:
            img_ref,img_dis1,img_dis2 = self.augm.transform_triplets(img_ref,img_dis1,img_dis2)

        if self.transform is not None:
            img_ref = self.transform(img_ref)
            img_dis1 = self.transform(img_dis1)
            img_dis2 = self.transform(img_dis2)
            if self.saliency:
                transformations = transforms.Compose([transforms.Resize(self.img_size),transforms.ToTensor()])
                saliency_map1 = transformations(saliency_map1)
                # saliency_map1 = (self.transform(saliency_map1)+1.)/2.
                saliency_map2 = transformations(saliency_map2)
                # saliency_map2 = (self.transform(saliency_map2)+1.)/2.

        y_mos1 = self.Y_train1.iloc[index]
        y_label1 = torch.FloatTensor(np.array(float(y_mos1/10)))    # transform to [0,1]
        y_mos2 = self.Y_train2.iloc[index]
        y_label2 = torch.FloatTensor(np.array(float(y_mos2/10)))    # transform to [0,1]

        judge_img = (y_label1-y_label2+1.)/2. # transform to [0,1]


        # return img_dis,img_ref,y_label
        if self.saliency:
            return {'p0': img_dis1, 'p1': img_dis2, 'ref': img_ref, 'judge': judge_img, 'y0': y_label1, 'y1': y_label2, 's1': saliency_map1, 's2': saliency_map2,
                'p0_path': img_dis1_dir, 'p1_path': img_dis2_dir, 'ref_path': img_ref_dir}
        else:
            return {'p0': img_dis1, 'p1': img_dis2, 'ref': img_ref, 'judge': judge_img, 'y0': y_label1, 'y1': y_label2,
                'p0_path': img_dis1_dir, 'p1_path': img_dis2_dir, 'ref_path': img_ref_dir}

    def __len__(self):
        return self.length



class FRIQA_AR_Dataset_test(Dataset):
    def __init__(self, data_dir, csv_path, transform, size=224, is_train=False, saliency=False):
        column_names = ['DisImg','RefImg','MOS']
        tmp_df = pd.read_csv(csv_path, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")
        self.data_dir = data_dir
        self.transform = transform
        self.img_size = size
        self.X_train_dis = tmp_df[['DisImg']]
        self.X_train_ref = tmp_df[['RefImg']]
        self.Y_train = tmp_df['MOS']

        self.length = len(tmp_df)

        self.saliency = saliency

        if is_train:
            self.augm = DataAugmentation(
                img_size=self.img_size,
                with_random_hflip=True,
                with_random_crop=True)
        else:
            self.augm = DataAugmentation(
                img_size=self.img_size,
                with_random_hflip=False,
                with_random_crop=False)

        # self.data_dir_dis = data_dir_dis
        # self.data_dir_ref = data_dir_ref
        # # self.dis = glob.glob(os.path.join(self.data_dir_dis, '*'+suff))
        # # self.ref = glob.glob(os.path.join(self.data_dir_ref, '*'+suff))
        # self.dis = []
        # self.ref = []
        # for i in range(300):
        #     self.dis.append(os.path.join(self.data_dir_dis,'{:06d}.png'.format(i+1)))
        #     self.ref.append(os.path.join(self.data_dir_ref,'{:06d}.png'.format(i+1)))

    def __getitem__(self, index):
        # this_dir = os.path.join(self.data_dir,self.X_train_dis.iloc[index,0])
        this_dir = self.data_dir+self.X_train_dis.iloc[index,0]
        # img_dis = Image.open(this_dir)
        # img_dis = img_dis.convert('RGB')
        img_dis = cv2.imread(this_dir, cv2.IMREAD_COLOR)
        img_dis = cv2.cvtColor(img_dis, cv2.COLOR_BGR2RGB)

        # this_dir = self.ref[index]
        # this_dir = os.path.join(self.data_dir,self.X_train_ref.iloc[index,0])
        this_dir = self.data_dir+self.X_train_ref.iloc[index,0]
        # img_ref = Image.open(this_dir)
        # img_ref = img_ref.convert('RGB')
        img_ref = cv2.imread(this_dir, cv2.IMREAD_COLOR)
        img_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2RGB)

        if self.saliency:
            saliency_map_dir = this_dir.replace('/img_AR/reference', '/img_AR_sal')
            saliency_map = cv2.imread(saliency_map_dir, cv2.IMREAD_COLOR)
            saliency_map = cv2.cvtColor(saliency_map, cv2.COLOR_BGR2GRAY)
            
        if self.saliency:
            img_dis,img_ref,saliency_map = self.augm.transform_triplets(img_dis, img_ref, saliency_map)
        else:
            img_dis,img_ref = self.augm.transform_twins(img_dis, img_ref)

        if self.transform is not None:
            img_dis = self.transform(img_dis)
            img_ref = self.transform(img_ref)
            if self.saliency:
                transformations = transforms.Compose([transforms.Resize(self.img_size),transforms.ToTensor()])
                saliency_map = transformations(saliency_map)

        

        y_mos = self.Y_train.iloc[index]
        y_label = torch.FloatTensor(np.array(float(y_mos)))

        if self.saliency:
            return img_dis,img_ref,saliency_map,y_label
        else:
            return img_dis,img_ref,y_label

    def __len__(self):
        return self.length


class FRIQA_ARBG_Dataset(Dataset):
    def __init__(self, data_dir, csv_path, transform, size=224, is_train=True, saliency=False):
        column_names = ['RefImg0','RefImg1','DisImg0','DisImg1','MOS0','MOS1']
        tmp_df = pd.read_csv(csv_path, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")
        self.data_dir = data_dir
        self.transform = transform
        self.img_size = size
        self.X_train_ref0 = tmp_df[['RefImg0']]
        self.X_train_ref1 = tmp_df[['RefImg1']]
        self.X_train_dis0 = tmp_df[['DisImg0']]
        self.X_train_dis1 = tmp_df[['DisImg1']]
        self.Y_train0 = tmp_df['MOS0']
        self.Y_train1 = tmp_df['MOS1']

        self.saliency = saliency

        self.length = len(tmp_df)

        if is_train:
            self.augm = DataAugmentation(
                img_size=self.img_size,
                with_random_hflip=True,
                with_random_crop=False,
                crop_scale = (0.8, 1.0),
                crop_ratio=(0.9, 1.1))
        else:
            self.augm = DataAugmentation(
                img_size=self.img_size,
                with_random_hflip=False,
                with_random_crop=False)


    def __getitem__(self, index):
        # this_dir = os.path.join(self.data_dir,self.X_train_dis.iloc[index,0])
        img_ref0_dir = self.data_dir+self.X_train_ref0.iloc[index,0]
        img_ref0 = cv2.imread(img_ref0_dir, cv2.IMREAD_COLOR)
        img_ref0 = cv2.cvtColor(img_ref0, cv2.COLOR_BGR2RGB)

        img_ref1_dir = self.data_dir+self.X_train_ref1.iloc[index,0]
        img_ref1 = cv2.imread(img_ref1_dir, cv2.IMREAD_COLOR)
        img_ref1 = cv2.cvtColor(img_ref1, cv2.COLOR_BGR2RGB)

        img_dis0_dir = self.data_dir+self.X_train_dis0.iloc[index,0]
        img_dis0 = cv2.imread(img_dis0_dir, cv2.IMREAD_COLOR)
        img_dis0 = cv2.cvtColor(img_dis0, cv2.COLOR_BGR2RGB)

        img_dis1_dir = self.data_dir+self.X_train_dis1.iloc[index,0]
        img_dis1 = cv2.imread(img_dis1_dir, cv2.IMREAD_COLOR)
        img_dis1 = cv2.cvtColor(img_dis1, cv2.COLOR_BGR2RGB)

        
        if self.saliency:
            saliency_map_dir = img_ref0_dir.replace('/img_AR/reference', '/img_AR_sal')
            saliency_map = cv2.imread(saliency_map_dir, cv2.IMREAD_COLOR)
            saliency_map = cv2.cvtColor(saliency_map, cv2.COLOR_BGR2GRAY)
            

        if self.saliency:
            img_ref0,img_ref1,img_dis0,img_dis1,saliency_map = self.augm.transform_quintuplets(img_ref0,img_ref1,img_dis0,img_dis1,saliency_map)
        else:
            img_ref0,img_ref1,img_dis0,img_dis1 = self.augm.transform_quadruplets(img_ref0,img_ref1,img_dis0,img_dis1)

        if self.transform is not None:
            img_ref0 = self.transform(img_ref0)
            img_ref1 = self.transform(img_ref1)
            img_dis0 = self.transform(img_dis0)
            img_dis1 = self.transform(img_dis1)
            if self.saliency:
                transformations = transforms.Compose([transforms.Resize(self.img_size),transforms.ToTensor()])
                saliency_map = transformations(saliency_map)

        y_mos0 = self.Y_train0.iloc[index]
        y_label0 = torch.FloatTensor(np.array(float(y_mos0/10)))    # transform to [0,1]
        y_mos1 = self.Y_train1.iloc[index]
        y_label1 = torch.FloatTensor(np.array(float(y_mos1/10)))    # transform to [0,1]

        judge_img = (y_label0-y_label1+1.)/2. # transform to [0,1]


        # return img_dis,img_ref,y_label
        if self.saliency:
            return {'p0': img_dis0, 'p1': img_dis1, 'ref0': img_ref0, 'ref1': img_ref1, 'judge': judge_img, 'y0': y_label0, 'y1': y_label1, 's0': saliency_map,
                'p0_path': img_dis0_dir, 'p1_path': img_dis1_dir, 'ref_path0': img_ref0_dir, 'ref_path1': img_ref1_dir}
        else:
            return {'p0': img_dis0, 'p1': img_dis1, 'ref0': img_ref0, 'ref1': img_ref1, 'judge': judge_img, 'y0': y_label0, 'y1': y_label1,
                'p0_path': img_dis0_dir, 'p1_path': img_dis1_dir, 'ref_path0': img_ref0_dir, 'ref_path1': img_ref1_dir}

    def __len__(self):
        return self.length


class FRIQA_ARBG_Dataset_test(Dataset):
    def __init__(self, data_dir, csv_path, transform, size=224, is_train=False, saliency=False):
        column_names = ['RefImg0','RefImg1','DisImg','MOS']
        tmp_df = pd.read_csv(csv_path, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")
        self.data_dir = data_dir
        self.transform = transform
        self.img_size = size
        self.X_test_dis = tmp_df[['DisImg']]
        self.X_test_ref0 = tmp_df[['RefImg0']]
        self.X_test_ref1 = tmp_df[['RefImg1']]
        self.Y_test = tmp_df['MOS']

        self.length = len(tmp_df)

        self.saliency = saliency

        if is_train:
            self.augm = DataAugmentation(
                img_size=self.img_size,
                with_random_hflip=True,
                with_random_crop=True)
        else:
            self.augm = DataAugmentation(
                img_size=self.img_size,
                with_random_hflip=False,
                with_random_crop=False)

    def __getitem__(self, index):
        this_dir = self.data_dir+self.X_test_dis.iloc[index,0]
        img_dis = cv2.imread(this_dir, cv2.IMREAD_COLOR)
        img_dis = cv2.cvtColor(img_dis, cv2.COLOR_BGR2RGB)

        this_dir = self.data_dir+self.X_test_ref1.iloc[index,0] # AR dir
        img_ref1 = cv2.imread(this_dir, cv2.IMREAD_COLOR)
        img_ref1 = cv2.cvtColor(img_ref1, cv2.COLOR_BGR2RGB)

        this_dir = self.data_dir+self.X_test_ref0.iloc[index,0] # BG dir
        img_ref0 = cv2.imread(this_dir, cv2.IMREAD_COLOR)
        img_ref0 = cv2.cvtColor(img_ref0, cv2.COLOR_BGR2RGB)

        if self.saliency:
            saliency_map_dir = this_dir.replace('/img_AR/reference', '/img_AR_sal')
            saliency_map = cv2.imread(saliency_map_dir, cv2.IMREAD_COLOR)
            saliency_map = cv2.cvtColor(saliency_map, cv2.COLOR_BGR2GRAY)
            
        if self.saliency:
            img_dis,img_ref0,img_ref1,saliency_map = self.augm.transform_quadruplets(img_dis, img_ref0, img_ref1, saliency_map)
        else:
            img_dis,img_ref0,img_ref1 = self.augm.transform_quadruplets(img_dis, img_ref0, img_ref1)

        if self.transform is not None:
            img_dis = self.transform(img_dis)
            img_ref0 = self.transform(img_ref0)
            img_ref1 = self.transform(img_ref1)
            if self.saliency:
                transformations = transforms.Compose([transforms.Resize(self.img_size),transforms.ToTensor()])
                saliency_map = transformations(saliency_map)

        

        y_mos = self.Y_test.iloc[index]
        y_label = torch.FloatTensor(np.array(float(y_mos)))

        if self.saliency:
            return {'p0': img_dis, 'ref0': img_ref0, 'ref1': img_ref1, 'y0': y_label, 's0': saliency_map}
        else:
            return {'p0': img_dis, 'ref0': img_ref0, 'ref1': img_ref1, 'y0': y_label}

    def __len__(self):
        return self.length






class cfiqa_test(Dataset):
    def __init__(self, data_dir_dis, data_dir_ref, data_dir_ref_sal, transform, suff='.png', is_train=True, saliency=False):
        self.data_dir_dis = data_dir_dis
        self.data_dir_ref = data_dir_ref
        self.data_dir_ref_sal = data_dir_ref_sal
        self.transform = transform
        self.img_size = 512

        self.X_dis = os.listdir(self.data_dir_dis)
        self.X_ref = os.listdir(self.data_dir_ref)
        self.X_ref_sal = os.listdir(self.data_dir_ref_sal)

        self.length = len(self.X_dis)

        self.saliency = saliency

        if is_train:
            self.augm = DataAugmentation(
                img_size=self.img_size,
                with_random_hflip=True,
                with_random_crop=True)
        else:
            self.augm = DataAugmentation(
                img_size=self.img_size,
                with_random_hflip=False,
                with_random_crop=False)

    def __getitem__(self, index):
        this_dir = os.path.join(self.data_dir_dis, self.X_dis[index])
        img_dis = cv2.imread(this_dir, cv2.IMREAD_COLOR)
        img_dis = cv2.cvtColor(img_dis, cv2.COLOR_BGR2RGB)

        this_dir = os.path.join(self.data_dir_ref, self.X_ref[index])
        img_ref = cv2.imread(this_dir, cv2.IMREAD_COLOR)
        img_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2RGB)

        if self.saliency:
            saliency_map_dir = os.path.join(self.data_dir_ref_sal, self.X_ref_sal[index])
            saliency_map = cv2.imread(saliency_map_dir, cv2.IMREAD_COLOR)
            saliency_map = cv2.cvtColor(saliency_map, cv2.COLOR_BGR2GRAY)
            
        if self.saliency:
            img_dis,img_ref,saliency_map = self.augm.transform_triplets(img_dis, img_ref, saliency_map)
        else:
            img_dis,img_ref = self.augm.transform_twins(img_dis, img_ref)

        if self.transform is not None:
            img_dis = self.transform(img_dis)
            img_ref = self.transform(img_ref)
            if self.saliency:
                transformations = transforms.Compose([transforms.Resize(224),transforms.ToTensor()])
                saliency_map = transformations(saliency_map)

        if self.saliency:
            return img_dis,img_ref,saliency_map
        else:
            return img_dis,img_ref

    def __len__(self):
        return self.length


class ariqa_test(Dataset):
    def __init__(self, data_dir_dis, data_dir_ref1, data_dir_ref1_sal, data_dir_ref2, transform, size=224, is_train=False, saliency=False):
        self.data_dir_dis = data_dir_dis
        self.data_dir_ref1 = data_dir_ref1
        self.data_dir_ref1_sal = data_dir_ref1_sal
        self.data_dir_ref2 = data_dir_ref2

        X_dis = os.listdir(self.data_dir_dis)
        X_dis = [name.zfill(7) for name in X_dis]
        X_dis = sorted(X_dis)
        self.X_dis = []
        for name in X_dis:
            temp_name = name.lstrip('0')
            self.X_dis.append(temp_name)
        X_ref1 = os.listdir(self.data_dir_ref1)
        X_ref1 = [name.zfill(7) for name in X_ref1]
        X_ref1 = sorted(X_ref1)
        self.X_ref1 = []
        for name in X_ref1:
            temp_name = name.lstrip('0')
            self.X_ref1.append(temp_name)
        # self.X_ref1 = os.listdir(self.data_dir_ref1)
        self.X_ref1_sal =self.X_ref1
        self.X_ref2 = self.X_ref1
        self.transform = transform
        self.img_size = size

        self.length = len(self.X_dis)

        self.saliency = saliency

        if is_train:
            self.augm = DataAugmentation(
                img_size=self.img_size,
                with_random_hflip=True,
                with_random_crop=True)
        else:
            self.augm = DataAugmentation(
                img_size=self.img_size,
                with_random_hflip=False,
                with_random_crop=False)

    def __getitem__(self, index):
        this_dir = os.path.join(self.data_dir_dis, self.X_dis[index])
        img_dis = cv2.imread(this_dir, cv2.IMREAD_COLOR)
        img_dis = cv2.cvtColor(img_dis, cv2.COLOR_BGR2RGB)

        this_dir = os.path.join(self.data_dir_ref1, self.X_ref1[(index%10)])
        img_ref0 = cv2.imread(this_dir, cv2.IMREAD_COLOR)
        img_ref0 = cv2.cvtColor(img_ref0, cv2.COLOR_BGR2RGB)

        if self.saliency:
            saliency_map_dir = os.path.join(self.data_dir_ref1_sal, self.X_ref1_sal[(index%10)])
            saliency_map = cv2.imread(saliency_map_dir, cv2.IMREAD_COLOR)
            saliency_map = cv2.cvtColor(saliency_map, cv2.COLOR_BGR2GRAY)

        this_dir = os.path.join(self.data_dir_ref2, self.X_ref2[(index%10)])
        img_ref1 = cv2.imread(this_dir, cv2.IMREAD_COLOR)
        img_ref1 = cv2.cvtColor(img_ref1, cv2.COLOR_BGR2RGB)


            
        if self.saliency:
            img_dis,img_ref0,img_ref1,saliency_map = self.augm.transform_quadruplets(img_dis, img_ref0, img_ref1, saliency_map)
        else:
            img_dis,img_ref0,img_ref1 = self.augm.transform_quadruplets(img_dis, img_ref0, img_ref1)

        if self.transform is not None:
            img_dis = self.transform(img_dis)
            img_ref0 = self.transform(img_ref0)
            img_ref1 = self.transform(img_ref1)
            if self.saliency:
                transformations = transforms.Compose([transforms.Resize(self.img_size),transforms.ToTensor()])
                saliency_map = transformations(saliency_map)

        y_mos = 0.0
        y_label = torch.FloatTensor(np.array(float(y_mos)))

        if self.saliency:
            return {'p0': img_dis, 'ref0': img_ref0, 'ref1': img_ref1, 'y0': y_label, 's0': saliency_map}
        else:
            return {'p0': img_dis, 'ref0': img_ref0, 'ref1': img_ref1, 'y0': y_label}

    def __len__(self):
        return self.length







# ---------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------

class FRIQA_AR_Dataset_v1(Dataset):
    def __init__(self, data_dir, csv_path, transform, suff='.png', is_train=True, saliency=False):
        column_names = ['RefImg','DisImg1','DisImg2','MOS1','MOS2']
        tmp_df = pd.read_csv(csv_path, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")
        self.data_dir = data_dir
        self.transform = transform
        self.img_size = 512
        self.X_train_ref = tmp_df[['RefImg']]
        self.X_train_dis1 = tmp_df[['DisImg1']]
        self.X_train_dis2 = tmp_df[['DisImg2']]
        self.Y_train1 = tmp_df['MOS1']
        self.Y_train2 = tmp_df['MOS2']

        self.saliency = saliency

        self.length = len(tmp_df)

        if is_train:
            self.augm = DataAugmentation(
                img_size=self.img_size,
                with_random_hflip=True,
                with_random_crop=False)
        else:
            self.augm = DataAugmentation(
                img_size=self.img_size,
                with_random_hflip=False,
                with_random_crop=False)


    def __getitem__(self, index):
        # this_dir = os.path.join(self.data_dir,self.X_train_dis.iloc[index,0])
        img_ref_dir = self.data_dir+self.X_train_ref.iloc[index,0]
        img_ref = cv2.imread(img_ref_dir, cv2.IMREAD_COLOR)
        img_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2RGB)

        img_dis1_dir = self.data_dir+self.X_train_dis1.iloc[index,0]
        img_dis1 = cv2.imread(img_dis1_dir, cv2.IMREAD_COLOR)
        img_dis1 = cv2.cvtColor(img_dis1, cv2.COLOR_BGR2RGB)

        img_dis2_dir = self.data_dir+self.X_train_dis2.iloc[index,0]
        img_dis2 = cv2.imread(img_dis2_dir, cv2.IMREAD_COLOR)
        img_dis2 = cv2.cvtColor(img_dis2, cv2.COLOR_BGR2RGB)

        
        if self.saliency:
            saliency_map1_dir = img_ref_dir.replace('/img_AR/reference', '/img_AR_sal')
            saliency_map1 = cv2.imread(saliency_map1_dir, cv2.IMREAD_COLOR)
            saliency_map1 = cv2.cvtColor(saliency_map1, cv2.COLOR_BGR2GRAY)
            saliency_map2_dir = img_ref_dir.replace('/img_AR/reference', '/img_AR_sal')
            saliency_map2 = cv2.imread(saliency_map2_dir, cv2.IMREAD_COLOR)
            saliency_map2 = cv2.cvtColor(saliency_map2, cv2.COLOR_BGR2GRAY)
            

        if self.saliency:
            img_ref,img_dis1,img_dis2,saliency_map1,saliency_map2 = self.augm.transform_quintuplets(img_ref,img_dis1,img_dis2,saliency_map1,saliency_map2)
        else:
            img_ref,img_dis1,img_dis2 = self.augm.transform_triplets(img_ref,img_dis1,img_dis2)

        if self.transform is not None:
            img_ref = self.transform(img_ref)
            img_dis1 = self.transform(img_dis1)
            img_dis2 = self.transform(img_dis2)
            if self.saliency:
                transformations = transforms.Compose([transforms.Resize(224),transforms.ToTensor()])
                saliency_map1 = transformations(saliency_map1)
                # saliency_map1 = (self.transform(saliency_map1)+1.)/2.
                saliency_map2 = transformations(saliency_map2)
                # saliency_map2 = (self.transform(saliency_map2)+1.)/2.

        y_mos1 = self.Y_train1.iloc[index]
        y_label1 = torch.FloatTensor(np.array(float(y_mos1/10)))    # transform to [0,1]
        y_mos2 = self.Y_train2.iloc[index]
        y_label2 = torch.FloatTensor(np.array(float(y_mos2/10)))    # transform to [0,1]

        judge_img = (y_label1-y_label2+1.)/2. # transform to [0,1]


        # return img_dis,img_ref,y_label
        if self.saliency:
            return {'p0': img_dis1, 'p1': img_dis2, 'ref': img_ref, 'judge': judge_img, 'y0': y_label1, 'y1': y_label2, 's1': saliency_map1, 's2': saliency_map2,
                'p0_path': img_dis1_dir, 'p1_path': img_dis2_dir, 'ref_path': img_ref_dir}
        else:
            return {'p0': img_dis1, 'p1': img_dis2, 'ref': img_ref, 'judge': judge_img, 'y0': y_label1, 'y1': y_label2,
                'p0_path': img_dis1_dir, 'p1_path': img_dis2_dir, 'ref_path': img_ref_dir}

    def __len__(self):
        return self.length



class FRIQA_AR_Dataset_test_v1(Dataset):
    def __init__(self, data_dir, csv_path, transform, suff='.png', is_train=True, saliency=False):
        column_names = ['DisImg','RefImg','MOS']
        tmp_df = pd.read_csv(csv_path, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")
        self.data_dir = data_dir
        self.transform = transform
        self.img_size = 512
        self.X_train_dis = tmp_df[['DisImg']]
        self.X_train_ref = tmp_df[['RefImg']]
        self.Y_train = tmp_df['MOS']

        self.length = len(tmp_df)

        self.saliency = saliency

        if is_train:
            self.augm = DataAugmentation(
                img_size=self.img_size,
                with_random_hflip=True,
                with_random_crop=True)
        else:
            self.augm = DataAugmentation(
                img_size=self.img_size,
                with_random_hflip=False,
                with_random_crop=False)

        # self.data_dir_dis = data_dir_dis
        # self.data_dir_ref = data_dir_ref
        # # self.dis = glob.glob(os.path.join(self.data_dir_dis, '*'+suff))
        # # self.ref = glob.glob(os.path.join(self.data_dir_ref, '*'+suff))
        # self.dis = []
        # self.ref = []
        # for i in range(300):
        #     self.dis.append(os.path.join(self.data_dir_dis,'{:06d}.png'.format(i+1)))
        #     self.ref.append(os.path.join(self.data_dir_ref,'{:06d}.png'.format(i+1)))

    def __getitem__(self, index):
        # this_dir = os.path.join(self.data_dir,self.X_train_dis.iloc[index,0])
        this_dir = self.data_dir+self.X_train_dis.iloc[index,0]
        # img_dis = Image.open(this_dir)
        # img_dis = img_dis.convert('RGB')
        img_dis = cv2.imread(this_dir, cv2.IMREAD_COLOR)
        img_dis = cv2.cvtColor(img_dis, cv2.COLOR_BGR2RGB)

        # this_dir = self.ref[index]
        # this_dir = os.path.join(self.data_dir,self.X_train_ref.iloc[index,0])
        this_dir = self.data_dir+self.X_train_ref.iloc[index,0]
        # img_ref = Image.open(this_dir)
        # img_ref = img_ref.convert('RGB')
        img_ref = cv2.imread(this_dir, cv2.IMREAD_COLOR)
        img_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2RGB)

        if self.saliency:
            saliency_map_dir = this_dir.replace('/img_AR/reference', '/img_AR_sal')
            saliency_map = cv2.imread(saliency_map_dir, cv2.IMREAD_COLOR)
            saliency_map = cv2.cvtColor(saliency_map, cv2.COLOR_BGR2GRAY)
            
        if self.saliency:
            img_dis,img_ref,saliency_map = self.augm.transform_triplets(img_dis, img_ref, saliency_map)
        else:
            img_dis,img_ref = self.augm.transform_twins(img_dis, img_ref)

        if self.transform is not None:
            img_dis = self.transform(img_dis)
            img_ref = self.transform(img_ref)
            if self.saliency:
                transformations = transforms.Compose([transforms.Resize(224),transforms.ToTensor()])
                saliency_map = transformations(saliency_map)

        

        y_mos = self.Y_train.iloc[index]
        y_label = torch.FloatTensor(np.array(float(y_mos)))

        if self.saliency:
            return img_dis,img_ref,saliency_map,y_label
        else:
            return img_dis,img_ref,y_label

    def __len__(self):
        return self.length


class FRIQA_ARBG_Dataset_v1(Dataset):
    def __init__(self, data_dir, csv_path, transform, suff='.png', is_train=True, saliency=False):
        column_names = ['RefImg0','RefImg1','DisImg0','DisImg1','MOS0','MOS1']
        tmp_df = pd.read_csv(csv_path, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")
        self.data_dir = data_dir
        self.transform = transform
        self.img_size = 512
        self.X_train_ref0 = tmp_df[['RefImg0']]
        self.X_train_ref1 = tmp_df[['RefImg1']]
        self.X_train_dis0 = tmp_df[['DisImg0']]
        self.X_train_dis1 = tmp_df[['DisImg1']]
        self.Y_train0 = tmp_df['MOS0']
        self.Y_train1 = tmp_df['MOS1']

        self.saliency = saliency

        self.length = len(tmp_df)

        if is_train:
            self.augm = DataAugmentation(
                img_size=self.img_size,
                with_random_hflip=True,
                with_random_crop=False)
        else:
            self.augm = DataAugmentation(
                img_size=self.img_size,
                with_random_hflip=False,
                with_random_crop=False)


    def __getitem__(self, index):
        # this_dir = os.path.join(self.data_dir,self.X_train_dis.iloc[index,0])
        img_ref0_dir = self.data_dir+self.X_train_ref0.iloc[index,0]
        img_ref0 = cv2.imread(img_ref0_dir, cv2.IMREAD_COLOR)
        img_ref0 = cv2.cvtColor(img_ref0, cv2.COLOR_BGR2RGB)

        img_ref1_dir = self.data_dir+self.X_train_ref1.iloc[index,0]
        img_ref1 = cv2.imread(img_ref1_dir, cv2.IMREAD_COLOR)
        img_ref1 = cv2.cvtColor(img_ref1, cv2.COLOR_BGR2RGB)

        img_dis0_dir = self.data_dir+self.X_train_dis0.iloc[index,0]
        img_dis0 = cv2.imread(img_dis0_dir, cv2.IMREAD_COLOR)
        img_dis0 = cv2.cvtColor(img_dis0, cv2.COLOR_BGR2RGB)

        img_dis1_dir = self.data_dir+self.X_train_dis1.iloc[index,0]
        img_dis1 = cv2.imread(img_dis1_dir, cv2.IMREAD_COLOR)
        img_dis1 = cv2.cvtColor(img_dis1, cv2.COLOR_BGR2RGB)

        
        if self.saliency:
            saliency_map_dir = img_ref0_dir.replace('/img_AR/reference', '/img_AR_sal')
            saliency_map = cv2.imread(saliency_map_dir, cv2.IMREAD_COLOR)
            saliency_map = cv2.cvtColor(saliency_map, cv2.COLOR_BGR2GRAY)
            

        if self.saliency:
            img_ref0,img_ref1,img_dis0,img_dis1,saliency_map = self.augm.transform_quintuplets(img_ref0,img_ref1,img_dis0,img_dis1,saliency_map)
        else:
            img_ref0,img_ref1,img_dis0,img_dis1 = self.augm.transform_quadruplets(img_ref0,img_ref1,img_dis0,img_dis1)

        if self.transform is not None:
            img_ref0 = self.transform(img_ref0)
            img_ref1 = self.transform(img_ref1)
            img_dis0 = self.transform(img_dis0)
            img_dis1 = self.transform(img_dis1)
            if self.saliency:
                transformations = transforms.Compose([transforms.Resize(224),transforms.ToTensor()])
                saliency_map = transformations(saliency_map)

        y_mos0 = self.Y_train0.iloc[index]
        y_label0 = torch.FloatTensor(np.array(float(y_mos0/10)))    # transform to [0,1]
        y_mos1 = self.Y_train1.iloc[index]
        y_label1 = torch.FloatTensor(np.array(float(y_mos1/10)))    # transform to [0,1]

        judge_img = (y_label0-y_label1+1.)/2. # transform to [0,1]


        # return img_dis,img_ref,y_label
        if self.saliency:
            return {'p0': img_dis0, 'p1': img_dis1, 'ref0': img_ref0, 'ref1': img_ref1, 'judge': judge_img, 'y0': y_label0, 'y1': y_label1, 's0': saliency_map,
                'p0_path': img_dis0_dir, 'p1_path': img_dis1_dir, 'ref_path0': img_ref0_dir, 'ref_path1': img_ref1_dir}
        else:
            return {'p0': img_dis0, 'p1': img_dis1, 'ref0': img_ref0, 'ref1': img_ref1, 'judge': judge_img, 'y0': y_label0, 'y1': y_label1,
                'p0_path': img_dis0_dir, 'p1_path': img_dis1_dir, 'ref_path0': img_ref0_dir, 'ref_path1': img_ref1_dir}

    def __len__(self):
        return self.length


class FRIQA_ARBG_Dataset_test_v1(Dataset):
    def __init__(self, data_dir, csv_path, transform, suff='.png', is_train=True, saliency=False):
        column_names = ['RefImg0','RefImg1','DisImg','MOS']
        tmp_df = pd.read_csv(csv_path, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")
        self.data_dir = data_dir
        self.transform = transform
        self.img_size = 512
        self.X_test_dis = tmp_df[['DisImg']]
        self.X_test_ref0 = tmp_df[['RefImg0']]
        self.X_test_ref1 = tmp_df[['RefImg1']]
        self.Y_test = tmp_df['MOS']

        self.length = len(tmp_df)

        self.saliency = saliency

        if is_train:
            self.augm = DataAugmentation(
                img_size=self.img_size,
                with_random_hflip=True,
                with_random_crop=True)
        else:
            self.augm = DataAugmentation(
                img_size=self.img_size,
                with_random_hflip=False,
                with_random_crop=False)

    def __getitem__(self, index):
        this_dir = self.data_dir+self.X_test_dis.iloc[index,0]
        img_dis = cv2.imread(this_dir, cv2.IMREAD_COLOR)
        img_dis = cv2.cvtColor(img_dis, cv2.COLOR_BGR2RGB)

        this_dir = self.data_dir+self.X_test_ref1.iloc[index,0] # AR dir
        img_ref1 = cv2.imread(this_dir, cv2.IMREAD_COLOR)
        img_ref1 = cv2.cvtColor(img_ref1, cv2.COLOR_BGR2RGB)

        this_dir = self.data_dir+self.X_test_ref0.iloc[index,0] # BG dir
        img_ref0 = cv2.imread(this_dir, cv2.IMREAD_COLOR)
        img_ref0 = cv2.cvtColor(img_ref0, cv2.COLOR_BGR2RGB)

        if self.saliency:
            saliency_map_dir = this_dir.replace('/img_AR/reference', '/img_AR_sal')
            saliency_map = cv2.imread(saliency_map_dir, cv2.IMREAD_COLOR)
            saliency_map = cv2.cvtColor(saliency_map, cv2.COLOR_BGR2GRAY)
            
        if self.saliency:
            img_dis,img_ref0,img_ref1,saliency_map = self.augm.transform_quadruplets(img_dis, img_ref0, img_ref1, saliency_map)
        else:
            img_dis,img_ref0,img_ref1 = self.augm.transform_quadruplets(img_dis, img_ref0, img_ref1)

        if self.transform is not None:
            img_dis = self.transform(img_dis)
            img_ref0 = self.transform(img_ref0)
            img_ref1 = self.transform(img_ref1)
            if self.saliency:
                transformations = transforms.Compose([transforms.Resize(224),transforms.ToTensor()])
                saliency_map = transformations(saliency_map)

        

        y_mos = self.Y_test.iloc[index]
        y_label = torch.FloatTensor(np.array(float(y_mos)))

        if self.saliency:
            return {'p0': img_dis, 'ref0': img_ref0, 'ref1': img_ref1, 'y0': y_label, 's0': saliency_map}
        else:
            return {'p0': img_dis, 'ref0': img_ref0, 'ref1': img_ref1, 'y0': y_label}

    def __len__(self):
        return self.length