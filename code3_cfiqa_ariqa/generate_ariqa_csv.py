## ------------------------------------------------------------------------------------------
## Confusing image quality assessment: Towards better augmented reality experience
## Huiyu Duan, Xiongkuo Min, Yucheng Zhu, Guangtao Zhai, Xiaokang Yang, and Patrick Le Callet
## IEEE Transactions on Image Processing (TIP)
## ------------------------------------------------------------------------------------------

import os
import pandas as pd
import scipy.io as scio
import math

from random import shuffle

if __name__ == '__main__':

    split = 2
    img_seq1 = [1,2,3,4,5,6,7,8]    # natural
    img_seq2 = [9,10,11,12,13,14,15,16] # webpage
    img_seq3 = [17,18,19,20]    # graphic
    shuffle(img_seq1)
    shuffle(img_seq2)
    shuffle(img_seq3)
    img_seq = img_seq1[0:int(len(img_seq1)/split)]+ \
        img_seq2[0:int(len(img_seq2)/split)]+ \
        img_seq3[0:int(len(img_seq3)/split)]+ \
        img_seq1[int(len(img_seq1)/split):int(len(img_seq1))]+ \
        img_seq2[int(len(img_seq2)/split):int(len(img_seq2))]+ \
        img_seq3[int(len(img_seq3)/split):int(len(img_seq3))]
    print(img_seq)

    write_csv_path = './train_test_split/ARIQA/'
    if not os.path.exists(write_csv_path):
        os.makedirs(write_csv_path)
    mat_file_name = '../database2_ariqa/MOS.mat'
    # load VRIQA mos from .mat file
    ARIQA_mos = scio.loadmat(mat_file_name)
    # the key of the mos is 'MOS'
    ARIQA_mos_value = ARIQA_mos['MOS']

    ## ----------------------------------------------------- train/test split -----------------------------------------------------
    # dataset organization
    disImgName = []
    refImgName = []
    refImgName2 = []
    mos = []
    mos2 = []

    for i in range(560):
        disImgName.append(str(i+1)+'.png')
        cnt = math.ceil(((i%140)+1)/7)
        refImgName.append(str(cnt)+'.png')
        mos.append(ARIQA_mos_value[i][0])

    # shuffle train sequnce
    for index_cross in range(split):
        train_seq = img_seq[0:int(len(img_seq)*index_cross/split)]+img_seq[int(len(img_seq)*(index_cross+1)/split):int(len(img_seq))]
        test_seq = img_seq[int(len(img_seq)*index_cross/split):int(len(img_seq)*(index_cross+1)/split)]
        train_dis_img1 = []
        train_dis_img2 = []
        train_ref_img1 = []
        train_ref_img2 = []
        train_mos1 = []
        train_mos2 = []
        test_dis_img = []
        test_ref_img1 = []
        test_ref_img2 = []
        test_mos = []
        for cnt in range(560):  # 20 ref * 7 distorted level * 4 mixed alpha level = 560
            ref_index = math.ceil(((cnt%140)+1)/7)
            if ref_index in train_seq:
                for i in range(4):  # 4 mixed alpha level
                    for j in range(7):  # 7 distorted images for each level
                        train_dis_img1.append('/img_Mixed/cropped/'+str(cnt+1)+'.png')
                        train_dis_img2.append('/img_Mixed/cropped/'+str(140*i+(ref_index-1)*7+j+1)+'.png')
                        train_ref_img1.append('/img_AR/reference/'+str(ref_index)+'.png')
                        train_ref_img2.append('/img_BG/cropped/'+str(ref_index)+'.png')
                        train_mos1.append(ARIQA_mos_value[cnt][0])
                        train_mos2.append(ARIQA_mos_value[140*i+(ref_index-1)*7+j][0])
            if ref_index in test_seq:
                test_dis_img.append('/img_Mixed/cropped/'+str(cnt+1)+'.png')
                test_ref_img1.append('/img_AR/reference/'+str(ref_index)+'.png')
                test_ref_img2.append('/img_BG/cropped/'+str(ref_index)+'.png')
                test_mos.append(ARIQA_mos_value[cnt][0])

        column_dataframe_train = ['RefImg0','RefImg1','DisImg0','DisImg1','MOS0','MOS1']
        column_dataframe = ['RefImg0','RefImg1','DisImg','MOS']
        train_image_dataframe = []
        test_image_dataframe = []
        for i_ref1,i_ref2,i_dis1,i_dis2,i_mos1,i_mos2 in zip(train_ref_img1,train_ref_img2,train_dis_img1,train_dis_img2,train_mos1,train_mos2):
            each_row = []
            each_row.append(i_ref1)
            each_row.append(i_ref2)
            each_row.append(i_dis1)
            each_row.append(i_dis2)
            each_row.append(i_mos1)    
            each_row.append(i_mos2)    
            train_image_dataframe.append(each_row)
        for i_ref1,i_ref2,i_dis,i_mos in zip(test_ref_img1,test_ref_img2,test_dis_img,test_mos):
            each_row = []
            each_row.append(i_ref1)
            each_row.append(i_ref2)
            each_row.append(i_dis)
            each_row.append(i_mos)  
            test_image_dataframe.append(each_row)

        # train
        train_df = pd.DataFrame(train_image_dataframe,columns = column_dataframe_train)
        train_df.to_csv(write_csv_path+'ARIQA_train_'+str(index_cross)+'.csv',header=False,index=False)
        # test
        test_df = pd.DataFrame(test_image_dataframe,columns = column_dataframe)
        test_df.to_csv(write_csv_path+'ARIQA_test_'+str(index_cross)+'.csv',header=False,index=False)

    ## ----------------------------------------------------- all -----------------------------------------------------
    # dataset organization
    disImgName = []
    refImgName1 = []
    refImgName2 = []
    mos = []
    mos2 = []
    for i in range(560):
        disImgName.append('/img_Mixed/cropped/'+str(i+1)+'.png')
        cnt = math.ceil(((i%140)+1)/7)
        refImgName1.append('/img_AR/reference/'+str(cnt)+'.png')
        refImgName2.append('/img_BG/cropped/'+str(cnt)+'.png')
        mos.append(ARIQA_mos_value[i][0])

    test_dis_img = disImgName
    test_ref_img1 = refImgName1
    test_ref_img2 = refImgName2
    test_mos = mos

    column_dataframe = ['RefImg0','RefImg1','DisImg','MOS']
    test_image_dataframe = []
    for i_ref1,i_ref2,i_dis,i_mos in zip(test_ref_img1,test_ref_img2,test_dis_img,test_mos):
        each_row = []
        each_row.append(i_ref1)
        each_row.append(i_ref2)
        each_row.append(i_dis)
        each_row.append(i_mos)  
        test_image_dataframe.append(each_row)

    # test
    test_df = pd.DataFrame(test_image_dataframe,columns = column_dataframe)
    test_df.to_csv(write_csv_path+'ARIQA_all'+'.csv',header=False,index=False)


    ## ----------------------------------------------------- baseline 1 -----------------------------------------------------
    disImgName = []
    refImgName = []
    mos = []
    for i in range(560):
        cnt = math.ceil(((i%140)+1)/7)
        cnt2 = (i%140)%7
        if (cnt2==1) | (cnt2==2):
            disImgName.append('/img_AR/distorted/'+str(cnt)+'_'+str(cnt2)+'.jpg')
        else:
            disImgName.append('/img_AR/distorted/'+str(cnt)+'_'+str(cnt2)+'.png')
        refImgName.append('/img_AR/reference/'+str(cnt)+'.png')
        mos.append(ARIQA_mos_value[i][0])

    test_dis_img = disImgName
    test_ref_img = refImgName
    test_mos = mos

    column_dataframe = ['DisImg','RefImg','MOS']
    test_image_dataframe = []
    for i_dis,i_ref,i_mos in zip(test_dis_img,test_ref_img,test_mos):
        each_row = []
        each_row.append(i_dis)
        each_row.append(i_ref)
        each_row.append(i_mos)  
        test_image_dataframe.append(each_row)

    # test
    test_df = pd.DataFrame(test_image_dataframe,columns = column_dataframe)
    test_df.to_csv(write_csv_path+'baseline1'+'.csv',header=False,index=False)

    ## ----------------------------------------------------- baseline a -----------------------------------------------------
    disImgName = []
    refImgName = []
    mos = []
    for i in range(560):
        disImgName.append('/img_Mixed/cropped/'+str(i+1)+'.png')
        cnt = math.ceil(((i%140)+1)/7)
        refImgName.append('/img_AR/reference/'+str(cnt)+'.png')
        mos.append(ARIQA_mos_value[i][0])

    test_dis_img = disImgName
    test_ref_img = refImgName
    test_mos = mos

    column_dataframe = ['DisImg','RefImg','MOS']
    test_image_dataframe = []
    for i_dis,i_ref,i_mos in zip(test_dis_img,test_ref_img,test_mos):
        each_row = []
        each_row.append(i_dis)
        each_row.append(i_ref)
        each_row.append(i_mos)  
        test_image_dataframe.append(each_row)

    # test
    test_df = pd.DataFrame(test_image_dataframe,columns = column_dataframe)
    test_df.to_csv(write_csv_path+'baseline_a'+'.csv',header=False,index=False)

    ## ----------------------------------------------------- baseline b -----------------------------------------------------
    disImgName = []
    refImgName = []
    mos = []
    for i in range(560):
        disImgName.append('/img_Mixed/cropped/'+str(i+1)+'.png')
        cnt = math.ceil(((i%140)+1)/7)
        refImgName.append('/img_BG/cropped/'+str(cnt)+'.png')
        mos.append(ARIQA_mos_value[i][0])

    test_dis_img = disImgName
    test_ref_img = refImgName
    test_mos = mos

    column_dataframe = ['DisImg','RefImg','MOS']
    test_image_dataframe = []
    for i_dis,i_ref,i_mos in zip(test_dis_img,test_ref_img,test_mos):
        each_row = []
        each_row.append(i_dis)
        each_row.append(i_ref)
        each_row.append(i_mos)  
        test_image_dataframe.append(each_row)

    # test
    test_df = pd.DataFrame(test_image_dataframe,columns = column_dataframe)
    test_df.to_csv(write_csv_path+'baseline_b'+'.csv',header=False,index=False)
