import argparse
import csv
import cv2
from glob import glob
import imageio
import numpy as np
import os
import pandas as pd
from PIL import Image


SIZE = (512,512)   # resized image size (width, height)
ratio = 0.8


def get_img_list(folder, ext='.jpg'):
    if ext is None:
        pattern = '*'
    else:
        pattern = '*' + ext
    return glob(os.path.join(folder, pattern))


def resize_background(src):
    # desired size
    dsize = SIZE
    # resize image
    return cv2.resize(src, dsize, interpolation = cv2.INTER_CUBIC)


def resize_augmented(src):
    h, w, c = src.shape
    # desired size
    if w/h > SIZE[0]/SIZE[1]:
        dsize = (int(SIZE[0]*ratio), int(h/w*SIZE[0]*ratio))
    else:
        dsize = (int(w/h*SIZE[1]*ratio), int(SIZE[1]*ratio))
    # resize image
    return cv2.resize(src, dsize, interpolation = cv2.INTER_CUBIC)


def resize_augmented2(src):
    h, w, c = src.shape
    # desired size
    dsize = (int(SIZE[0]*ratio),int(SIZE[1]*ratio))
    # resize image
    return cv2.resize(src, dsize, interpolation = cv2.INTER_CUBIC)


def merge(img1, img2, beta):
    return cv2.addWeighted(img1, 1 - beta, img2, beta, 0)


def generate_images(opt):
    A_dir = os.path.join(opt.dataroot, opt.augmented_folder) # folder for augmented images
    B_dir = os.path.join(opt.dataroot, opt.background_folder) # folder for background images
    A_list = get_img_list(A_dir,ext='jpg')
    B_list = get_img_list(B_dir,ext='jpg')
    out_dir = os.path.join(opt.outf)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    if not os.path.exists(os.path.join(out_dir, 'M')):
        os.mkdir(os.path.join(out_dir, 'M'))
    if not os.path.exists(os.path.join(out_dir, 'A')):
        os.mkdir(os.path.join(out_dir, 'A'))
    if not os.path.exists(os.path.join(out_dir, 'B')):
        os.mkdir(os.path.join(out_dir, 'B'))
    i = 0
    n = 0
    whole_dataframe = []
    for A_root in A_list:
        # for B_root in B_list:
        B_root = B_list[n]
        n += 1
        # THRESHOLDS = [round(random.uniform(0.05, 0.95), 2)]
        THRESHOLDS = [round(np.random.beta(5, 5), 2)]
        for thres in THRESHOLDS:
            # thres = THRESHOLDS[i]
            print(str(i+1)+': '+str(thres))
            i += 1
            A = np.array(Image.open(A_root).convert('RGB'))
            B = np.array(Image.open(B_root).convert('RGB'))
            A_resize = resize_background(A)
            B_reszie = resize_background(B)
            I = merge(B_reszie,A_resize,thres)
            # write image
            imageio.imwrite(os.path.join(out_dir, 'M', '{:06d}.png'.format(i)), I)
            imageio.imwrite(os.path.join(out_dir, 'A', '{:06d}.png'.format(i)), A_resize)
            imageio.imwrite(os.path.join(out_dir, 'B', '{:06d}.png'.format(i)), B_reszie)
            # write thresholds
            each_row = []
            each_row.append(i)
            each_row.append('{:06d}.png'.format(i))
            each_row.append(thres)
            whole_dataframe.append(each_row)
    
    column_dataframe = ['id','img_name','thresholds']
    whole_df = pd.DataFrame(whole_dataframe,columns = column_dataframe)
    whole_df.to_csv(os.path.join(out_dir,'thresholds.csv'),header=False,index=False)


def generate_images_from_csv(opt):
    A_dir = os.path.join(opt.dataroot, opt.augmented_folder) # folder for augmented images
    B_dir = os.path.join(opt.dataroot, opt.background_folder) # folder for background images
    A_list = get_img_list(A_dir,ext='jpg')
    B_list = get_img_list(B_dir,ext='jpg')
    out_dir = os.path.join(opt.outf)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    if not os.path.exists(os.path.join(out_dir, 'M')):
        os.mkdir(os.path.join(out_dir, 'M'))
    if not os.path.exists(os.path.join(out_dir, 'A')):
        os.mkdir(os.path.join(out_dir, 'A'))
    if not os.path.exists(os.path.join(out_dir, 'B')):
        os.mkdir(os.path.join(out_dir, 'B'))
    i = 0
    THRESHOLDS = []
    if os.path.isfile(os.path.join(out_dir,'thresholds.csv')):
        with open(os.path.join(out_dir,'thresholds.csv'), newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                THRESHOLDS.append(float(row[2]))
    for A_root in A_list:
        # for B_root in B_list:
        B_root = B_list[i]
        thres = THRESHOLDS[i]
        print(str(i+1)+': '+str(thres))
        i += 1
        A = np.array(Image.open(A_root).convert('RGB'))
        B = np.array(Image.open(B_root).convert('RGB'))
        A_resize = resize_background(A)
        B_reszie = resize_background(B)
        I = merge(B_reszie,A_resize,thres)
        # write image
        imageio.imwrite(os.path.join(out_dir, 'M', '{:06d}.png'.format(i)), I)
        imageio.imwrite(os.path.join(out_dir, 'A', '{:06d}.png'.format(i)), A_resize)
        imageio.imwrite(os.path.join(out_dir, 'B', '{:06d}.png'.format(i)), B_reszie)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', required=True, help='path to raw reference images')
    parser.add_argument('--background_folder', required=True, help='folder of background raw reference images')
    parser.add_argument('--augmented_folder', required=True, help='folder of augmented raw reference images')
    parser.add_argument('--outf', required=True, help='folder to output generated dataset')
    opt = parser.parse_args()
    print(opt)

    generate_images(opt)
    # generate_images_from_csv(opt)