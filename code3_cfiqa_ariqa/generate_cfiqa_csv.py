## ------------------------------------------------------------------------------------------
## Confusing image quality assessment: Towards better augmented reality experience
## Huiyu Duan, Xiongkuo Min, Yucheng Zhu, Guangtao Zhai, Xiaokang Yang, and Patrick Le Callet
## IEEE Transactions on Image Processing (TIP)
## ------------------------------------------------------------------------------------------

import os
import pandas as pd
import scipy.io as scio

if __name__ == '__main__':

    write_csv_path = './train_test_split/CFIQA/'
    if not os.path.exists(write_csv_path):
        os.makedirs(write_csv_path)
    mat_file_name = '../database1_cfiqa/MOS.mat'
    # load VRIQA mos from .mat file
    CFIQA_mos = scio.loadmat(mat_file_name)
    # the key of the mos is 'MOS'
    CFIQA_mos_value = CFIQA_mos['MOS']

    # dataset organization
    disImgName = []
    refImgName = []
    refImgName2 = []
    mos = []
    mos2 = []
    split = 2

    for i in range(300):
        disImgName.append('/M/'+'{:06d}.png'.format(i+1))
        refImgName.append('/A/'+'{:06d}.png'.format(i+1))
        mos.append(CFIQA_mos_value[i][0])
        refImgName2.append('/B/'+'{:06d}.png'.format(i+1))
        mos2.append(CFIQA_mos_value[i+300][0])

    # split the cfiqa database into training and evaluation sets (2-fold cross validation)
    for index_cross in range(split):

        train_dis_img = disImgName[0:int(len(disImgName)*index_cross/split)]+disImgName[int(len(disImgName)*(index_cross+1)/split):int(len(disImgName))]
        test_dis_img = disImgName[int(len(disImgName)*index_cross/split):int(len(disImgName)*(index_cross+1)/split)]
        train_ref_img = refImgName[0:int(len(refImgName)*index_cross/split)]+refImgName[int(len(refImgName)*(index_cross+1)/split):int(len(refImgName))]
        test_ref_img = refImgName[int(len(refImgName)*index_cross/split):int(len(refImgName)*(index_cross+1)/split)]
        train_ref_img2 = refImgName2[0:int(len(refImgName2)*index_cross/split)]+refImgName2[int(len(refImgName2)*(index_cross+1)/split):int(len(refImgName2))]
        test_ref_img2 = refImgName2[int(len(refImgName2)*index_cross/split):int(len(refImgName2)*(index_cross+1)/split)]
        train_mos = mos[0:int(len(mos)*index_cross/split)]+mos[int(len(mos)*(index_cross+1)/split):int(len(mos))]
        test_mos = mos[int(len(mos)*index_cross/split):int(len(mos)*(index_cross+1)/split)]
        train_mos2 = mos2[0:int(len(mos2)*index_cross/split)]+mos2[int(len(mos2)*(index_cross+1)/split):int(len(mos2))]
        test_mos2 = mos2[int(len(mos2)*index_cross/split):int(len(mos2)*(index_cross+1)/split)]

        column_dataframe = ['DisImg','RefImg','MOS']
        train_image_dataframe = []
        test_image_dataframe = []
        for i_dis,i_ref,i_ref2,i_mos,i_mos2 in zip(train_dis_img,train_ref_img,train_ref_img2,train_mos,train_mos2):
            each_row = []
            each_row.append(i_dis)
            each_row.append(i_ref)
            each_row.append(i_mos)    
            train_image_dataframe.append(each_row)
            each_row = []
            each_row.append(i_dis)
            each_row.append(i_ref2)
            each_row.append(i_mos2)    
            train_image_dataframe.append(each_row)
        for i_dis,i_ref,i_ref2,i_mos,i_mos2 in zip(test_dis_img,test_ref_img,test_ref_img2,test_mos,test_mos2):
            each_row = []
            each_row.append(i_dis)
            each_row.append(i_ref)
            each_row.append(i_mos)  
            test_image_dataframe.append(each_row)
            each_row = []
            each_row.append(i_dis)
            each_row.append(i_ref2)
            each_row.append(i_mos2)  
            test_image_dataframe.append(each_row)

        # train
        train_df = pd.DataFrame(train_image_dataframe,columns = column_dataframe)
        train_df.to_csv(write_csv_path+'CFIQA_train_'+str(index_cross)+'.csv',header=False,index=False)
        # test
        test_df = pd.DataFrame(test_image_dataframe,columns = column_dataframe)
        test_df.to_csv(write_csv_path+'CFIQA_test_'+str(index_cross)+'.csv',header=False,index=False)

    
    ## all images in the cfiqa database
    all_dis_img = disImgName
    all_ref_img = refImgName
    all_ref_img2 = refImgName2
    all_mos = mos
    all_mos2 = mos2

    column_dataframe = ['DisImg','RefImg','MOS']
    all_image_dataframe = []
    for i_dis,i_ref,i_ref2,i_mos,i_mos2 in zip(all_dis_img,all_ref_img,all_ref_img2,all_mos,all_mos2):
        each_row = []
        each_row.append(i_dis)
        each_row.append(i_ref)
        each_row.append(i_mos)  
        all_image_dataframe.append(each_row)
        each_row = []
        each_row.append(i_dis)
        each_row.append(i_ref2)
        each_row.append(i_mos2)  
        all_image_dataframe.append(each_row)

    # all
    all_df = pd.DataFrame(all_image_dataframe,columns = column_dataframe)
    all_df.to_csv(write_csv_path+'CFIQA_all'+'.csv',header=False,index=False)