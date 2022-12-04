% ------------------------------------------------------------------------------------------
% Confusing image quality assessment: Towards better augmented reality experience
% Huiyu Duan, Xiongkuo Min, Yucheng Zhu, Guangtao Zhai, Xiaokang Yang, and Patrick Le Callet
% IEEE Transactions on Image Processing (TIP)
% ------------------------------------------------------------------------------------------

clc
close all
clear all

%% performance

addpath('.\performance\')
addpath('.\PWRC')

MOS = load('..\database1_cfiqa\MOS.mat').MOS;
SD = load('..\database1_cfiqa\SD.mat').SD;
MOSz = load('..\database1_cfiqa\MOSz.mat').MOSz;
GT = MOSz;

confusion_IQA_FR = load('..\code2_baselines\cfiqa_FR_traditional_sal.mat');
confusion_IQA_FR_names = fieldnames(confusion_IQA_FR);

alpha1 = 0.3;
alpha2 = 0.7;

alpha3 = 0.4;
alpha4 = 0.6;

%% FR Algorithms
for i = 1:size(confusion_IQA_FR_names,1)
    temp_score = real(getfield(confusion_IQA_FR,confusion_IQA_FR_names{i,1}));
    score_FR(:,i) = [temp_score(:,1);temp_score(:,2)];
    
    perSRCC_FR(1,i) = corr(score_FR(:,i),GT,'type','Spearman');
    perKRCC_FR(1,i) = corr(score_FR(:,i),GT,'type','Kendall');
    [delta,beta,yhat,y,diff] = findrmse2(score_FR(:,i),GT);
    score_mapped(:,i) = yhat;
    perRMSE_FR(1,i) = sqrt(sum(diff.^2)/length(diff));
    perPLCC_FR(1,i) = corr(GT, yhat, 'type','Pearson');
    perPWRC_FR(1,i) = calPWRC(score_FR(:,i), MOS, SD);
end

%% after split to two sub-datasets

alpha_path = '..\database1_cfiqa\';
[numeric,text,raw] = xlsread([alpha_path,'thresholds.csv']);
alpha_data = numeric;
alpha(:,1) = alpha_data(:,3);
% choose alpha data range (alpha:(alpha1,alpha2))
for i = 1:300
    if (alpha(i,1)>alpha1)&&(alpha(i,1)<alpha2)
        DistortionType(i,1) = 1;
        DistortionType(i+300,1) = 1;
    else
        DistortionType(i,1) = 2;
        DistortionType(i+300,1) = 2;
    end
end
for i = 1:300
    if (alpha(i,1)>alpha3)&&(alpha(i,1)<alpha4)
        DistortionType2(i,1) = 1;
        DistortionType2(i+300,1) = 1;
    else
        DistortionType2(i,1) = 2;
        DistortionType2(i+300,1) = 2;
    end
end

% split the dataset to two sub-datasets (image:[1,150][151,300])
for i = 1:300
    if (i<=150)
        DatasetType(i,1) = 1;
        DatasetType(i+300,1) = 1;
    else
        DatasetType(i,1) = 2;
        DatasetType(i+300,1) = 2;
    end
end

for i = 1:size(confusion_IQA_FR_names,1)
    for j = 1:2     % four types
        scoreNewTemp = score_FR(:,i);
        scoreNew = scoreNewTemp(find(DatasetType==j));
        GTNew = GT(find(DatasetType==j));
        MOSNew = MOS(find(DatasetType==j));
        SDNew = SD(find(DatasetType==j));
        perSRCC_FR(j+1,i) = corr(scoreNew,GTNew,'type','Spearman');
        perKRCC_FR(j+1,i) = corr(scoreNew,GTNew,'type','Kendall');
        [delta,beta,yhat,y,diff] = findrmse2(scoreNew,GTNew);
        perRMSE_FR(j+1,i) = sqrt(sum(diff.^2)/length(diff));
        perPLCC_FR(j+1,i) = corr(GTNew, yhat, 'type','Pearson');
        perPWRC_FR(j+1,i) = calPWRC(scoreNew, MOSNew, SDNew);
    end
    
    for j = 1:2     % four types
        scoreNewTemp = score_FR(:,i);
        scoreNew = scoreNewTemp(find(DistortionType==1&DatasetType==j));
        GTNew = GT(find(DistortionType==1&DatasetType==j));
        MOSNew = MOS(find(DistortionType==1&DatasetType==j));
        SDNew = SD(find(DistortionType==1&DatasetType==j));
        perSRCC_FR(j+3,i) = corr(scoreNew,GTNew,'type','Spearman');
        perKRCC_FR(j+3,i) = corr(scoreNew,GTNew,'type','Kendall');
        [delta,beta,yhat,y,diff] = findrmse2(scoreNew,GTNew);
        perRMSE_FR(j+3,i) = sqrt(sum(diff.^2)/length(diff));
        perPLCC_FR(j+3,i) = corr(GTNew, yhat, 'type','Pearson');
        perPWRC_FR(j+3,i) = calPWRC(scoreNew, MOSNew, SDNew);
    end
    
    for j = 1:2     % four types
        scoreNewTemp = score_FR(:,i);
        scoreNew = scoreNewTemp(find(DistortionType2==1&DatasetType==j));
        GTNew = GT(find(DistortionType2==1&DatasetType==j));
        MOSNew = MOS(find(DistortionType2==1&DatasetType==j));
        SDNew = SD(find(DistortionType2==1&DatasetType==j));
        perSRCC_FR(j+5,i) = corr(scoreNew,GTNew,'type','Spearman');
        perKRCC_FR(j+5,i) = corr(scoreNew,GTNew,'type','Kendall');
        [delta,beta,yhat,y,diff] = findrmse2(scoreNew,GTNew);
        perRMSE_FR(j+5,i) = sqrt(sum(diff.^2)/length(diff));
        perPLCC_FR(j+5,i) = corr(GTNew, yhat, 'type','Pearson');
        perPWRC_FR(j+5,i) = calPWRC(scoreNew, MOSNew, SDNew);
    end
end

% Entire dataset
perSRCC_FR(8,:) = (perSRCC_FR(2,:)+perSRCC_FR(3,:))/2;
perKRCC_FR(8,:) = (perKRCC_FR(2,:)+perKRCC_FR(3,:))/2;
perRMSE_FR(8,:) = (perRMSE_FR(2,:)+perRMSE_FR(3,:))/2;
perPLCC_FR(8,:) = (perPLCC_FR(2,:)+perPLCC_FR(3,:))/2;
perPWRC_FR(8,:) = (perPWRC_FR(2,:)+perPWRC_FR(3,:))/2;

% 0.3<alpha<0.7
perSRCC_FR(9,:) = (perSRCC_FR(4,:)+perSRCC_FR(5,:))/2;
perKRCC_FR(9,:) = (perKRCC_FR(4,:)+perKRCC_FR(5,:))/2;
perRMSE_FR(9,:) = (perRMSE_FR(4,:)+perRMSE_FR(5,:))/2;
perPLCC_FR(9,:) = (perPLCC_FR(4,:)+perPLCC_FR(5,:))/2;
perPWRC_FR(9,:) = (perPWRC_FR(4,:)+perPWRC_FR(5,:))/2;

% 0.4<alpha<0.6
perSRCC_FR(10,:) = (perSRCC_FR(6,:)+perSRCC_FR(7,:))/2;
perKRCC_FR(10,:) = (perKRCC_FR(6,:)+perKRCC_FR(7,:))/2;
perRMSE_FR(10,:) = (perRMSE_FR(6,:)+perRMSE_FR(7,:))/2;
perPLCC_FR(10,:) = (perPLCC_FR(6,:)+perPLCC_FR(7,:))/2;
perPWRC_FR(10,:) = (perPWRC_FR(6,:)+perPWRC_FR(7,:))/2;