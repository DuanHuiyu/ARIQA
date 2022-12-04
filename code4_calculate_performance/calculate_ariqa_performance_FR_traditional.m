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

MOS = load('..\database2_ariqa\MOS.mat').MOS;
SD = load('..\database2_ariqa\SD.mat').SD;
MOSz = load('..\database2_ariqa\MOSz.mat').MOSz;
GT = MOSz;

%% FR Algorithms
%% Type I (AR_dis*alpha with AR_ref)
confusion_IQA_FR = load('..\code2_baselines\ariqa_FR_traditional_1.mat');
confusion_IQA_FR_names = fieldnames(confusion_IQA_FR);
for i = 1:size(confusion_IQA_FR_names,1)
    temp_score = real(getfield(confusion_IQA_FR,confusion_IQA_FR_names{i,1}));
    score_FR(:,i) = temp_score(:,1);
    
    perSRCC_FR(1,i) = corr(score_FR(:,i),GT,'type','Spearman');
    perKRCC_FR(1,i) = corr(score_FR(:,i),GT,'type','Kendall');
    [delta,beta,yhat,y,diff] = findrmse2(score_FR(:,i),GT);
    score_mapped(:,i) = yhat;
    perRMSE_FR(1,i) = sqrt(sum(diff.^2)/length(diff));
    perPLCC_FR(1,i) = corr(GT, yhat, 'type','Pearson');
    perPWRC_FR(1,i) = calPWRC(score_FR(:,i), MOS, SD);
end

%% Type II (Mixed with AR_ref)
confusion_IQA_FR = load('..\code2_baselines\ariqa_FR_traditional_a.mat');  % with raw reference
confusion_IQA_FR_names = fieldnames(confusion_IQA_FR);
for i = 1:size(confusion_IQA_FR_names,1)
    temp_score = real(getfield(confusion_IQA_FR,confusion_IQA_FR_names{i,1}));
    score_FR(:,i) = temp_score(:,1);
    
    perSRCC_FR(2,i) = corr(score_FR(:,i),GT,'type','Spearman');
    perKRCC_FR(2,i) = corr(score_FR(:,i),GT,'type','Kendall');
    [delta,beta,yhat,y,diff] = findrmse2(score_FR(:,i),GT);
    score_mapped(:,i) = yhat;
    perRMSE_FR(2,i) = sqrt(sum(diff.^2)/length(diff));
    perPLCC_FR(2,i) = corr(GT, yhat, 'type','Pearson');
    perPWRC_FR(2,i) = calPWRC(score_FR(:,i), MOS, SD);
end

%% webpage
score_FR_new = score_FR(1:224,:);
GT_new = GT(1:224);
MOS_new = MOS(1:224);
SD_new = SD(1:224);
for i = 1:size(confusion_IQA_FR_names,1)
    perSRCC_FR(3,i) = corr(score_FR_new(:,i),GT_new,'type','Spearman');
    perKRCC_FR(3,i) = corr(score_FR_new(:,i),GT_new,'type','Kendall');
    [delta,beta,yhat,y,diff] = findrmse2(score_FR_new(:,i),GT_new);
    perRMSE_FR(3,i) = sqrt(sum(diff.^2)/length(diff));
    perPLCC_FR(3,i) = corr(GT_new, yhat, 'type','Pearson');
    perPWRC_FR(3,i) = calPWRC(score_FR_new(:,i), MOS_new, SD_new);
end

%% natural
score_FR_new = score_FR(225:448,:);
GT_new = GT(225:448);
MOS_new = MOS(225:448);
SD_new = SD(225:448);
for i = 1:size(confusion_IQA_FR_names,1)
    perSRCC_FR(4,i) = corr(score_FR_new(:,i),GT_new,'type','Spearman');
    perKRCC_FR(4,i) = corr(score_FR_new(:,i),GT_new,'type','Kendall');
    [delta,beta,yhat,y,diff] = findrmse2(score_FR_new(:,i),GT_new);
    perRMSE_FR(4,i) = sqrt(sum(diff.^2)/length(diff));
    perPLCC_FR(4,i) = corr(GT_new, yhat, 'type','Pearson');
    perPWRC_FR(4,i) = calPWRC(score_FR_new(:,i), MOS_new, SD_new);
end

%% graphic
score_FR_new = score_FR(449:560,:);
GT_new = GT(449:560);
MOS_new = MOS(449:560);
SD_new = SD(449:560);
for i = 1:size(confusion_IQA_FR_names,1)
    perSRCC_FR(5,i) = corr(score_FR_new(:,i),GT_new,'type','Spearman');
    perKRCC_FR(5,i) = corr(score_FR_new(:,i),GT_new,'type','Kendall');
    [delta,beta,yhat,y,diff] = findrmse2(score_FR_new(:,i),GT_new);
    perRMSE_FR(5,i) = sqrt(sum(diff.^2)/length(diff));
    perPLCC_FR(5,i) = corr(GT_new, yhat, 'type','Pearson');
    perPWRC_FR(5,i) = calPWRC(score_FR_new(:,i), MOS_new, SD_new);
end

%% substract
confusion_IQA_FR_ar = load('..\code2_baselines\ariqa_FR_traditional_a.mat');  % with raw reference
confusion_IQA_FR_bg = load('..\code2_baselines\ariqa_FR_traditional_b.mat');  % with raw reference
confusion_IQA_FR_names_a = fieldnames(confusion_IQA_FR_ar);
confusion_IQA_FR_names_b = fieldnames(confusion_IQA_FR_bg);
for i = 1:size(confusion_IQA_FR_names,1)
    temp_score_a = real(getfield(confusion_IQA_FR_ar,confusion_IQA_FR_names_a{i,1}));
    temp_score_b = real(getfield(confusion_IQA_FR_bg,confusion_IQA_FR_names_b{i,1}));
    score_FR(:,i) = temp_score_a(:,1)-temp_score_b(:,1);
    
    perSRCC_FR(6,i) = corr(score_FR(:,i),GT,'type','Spearman');
    perKRCC_FR(6,i) = corr(score_FR(:,i),GT,'type','Kendall');
    [delta,beta,yhat,y,diff] = findrmse2(score_FR(:,i),GT);
    score_mapped(:,i) = yhat;
    perRMSE_FR(6,i) = sqrt(sum(diff.^2)/length(diff));
    perPLCC_FR(6,i) = corr(GT, yhat, 'type','Pearson');
    perPWRC_FR(6,i) = calPWRC(score_FR(:,i), MOS, SD);
end

%% dot division
confusion_IQA_FR_ar = load('..\code2_baselines\ariqa_FR_traditional_a.mat');  % with raw reference
confusion_IQA_FR_bg = load('..\code2_baselines\ariqa_FR_traditional_b.mat');  % with raw reference
confusion_IQA_FR_names_a = fieldnames(confusion_IQA_FR_ar);
confusion_IQA_FR_names_b = fieldnames(confusion_IQA_FR_bg);
for i = 1:size(confusion_IQA_FR_names,1)
    temp_score_a = real(getfield(confusion_IQA_FR_ar,confusion_IQA_FR_names_a{i,1}));
    temp_score_b = real(getfield(confusion_IQA_FR_bg,confusion_IQA_FR_names_b{i,1}));
    score_FR(:,i) = temp_score_a(:,1)./temp_score_b(:,1);
    
    perSRCC_FR(7,i) = corr(score_FR(:,i),GT,'type','Spearman');
    perKRCC_FR(7,i) = corr(score_FR(:,i),GT,'type','Kendall');
    [delta,beta,yhat,y,diff] = findrmse2(score_FR(:,i),GT);
    score_mapped(:,i) = yhat;
    perRMSE_FR(7,i) = sqrt(sum(diff.^2)/length(diff));
    perPLCC_FR(7,i) = corr(GT, yhat, 'type','Pearson');
    perPWRC_FR(7,i) = calPWRC(score_FR(:,i), MOS, SD);
end

%% Type III (SVR)
cd('SVM');
for i = 1:size(confusion_IQA_FR_names,1)
    score_FR_a(:,i) = real(getfield(confusion_IQA_FR_ar,confusion_IQA_FR_names_a{i,1}));
    score_FR_b(:,i) = real(getfield(confusion_IQA_FR_bg,confusion_IQA_FR_names_b{i,1}));
end
for cnt = 1:size(confusion_IQA_FR_names,1)
    clc
    disp('SVR');
    cnt
    
    Feature = [score_FR_a(:,cnt),score_FR_b(:,cnt)];
    img = 1:560;
    img = img';

    nSplit = 100;
    nStimuliRef = max(img);
    nTrainSample = round(nStimuliRef*0.8);
    for iSplit = 1:nSplit
        splits(iSplit).ind = randperm(nStimuliRef)';
        splits(iSplit).train = [];
        splits(iSplit).test = [];
        for iRef = 1:nTrainSample
            splits(iSplit).train = [splits(iSplit).train; find(img==splits(iSplit).ind(iRef))];
        end
        for iRef = nTrainSample+1:nStimuliRef
            splits(iSplit).test = [splits(iSplit).test; find(img==splits(iSplit).ind(iRef))];
        end 
    end

    fid = fopen('range','w');
    fprintf(fid,'x\n-1 1\n');
    for i = 1:size(Feature,2)
        fprintf(fid,'%d %f %f\n',i,min(Feature(:,i)),max(Feature(:,i)));
    end
    fclose(fid);

    for iSplit = 1:nSplit
        iSplit

        % training and testing
        % train
        FeatureTrain = Feature(splits(iSplit).train,:);
        GT_train = GT(splits(iSplit).train,:);
    %     LIVE_dmos_train = LIVE_vif(splits(iSplit).train,:);
        fid = fopen('train_ind.txt','w');
        for itr_im = 1:size(FeatureTrain,1)
            fprintf(fid,'%d ',GT_train(itr_im));
            for itr_param = 1:size(FeatureTrain,2)
                fprintf(fid,'%d:%f ',itr_param,FeatureTrain(itr_im,itr_param));
            end
            fprintf(fid,'\n');
        end
        fclose(fid);
        if(exist('train_scale','file'))
            delete train_scale
        end
        system('svm-scale -l -1 -u 1 -s range train_ind.txt > train_scale');
        system('svm-train -b 1 -s 3 -g 0.05 -c 1024 -q train_scale model');

        % test
        FeatureTest = Feature(splits(iSplit).test,:);
        mos=ones(size(FeatureTest,1),1);
        fid = fopen('test_ind.txt','w');
        for itr_im = 1:size(FeatureTest,1)
            fprintf(fid,'%d ',mos(itr_im));
            for itr_param = 1:size(FeatureTest,2)
                fprintf(fid,'%d:%f ',itr_param,FeatureTest(itr_im,itr_param));
            end
            fprintf(fid,'\n');
        end
        fclose(fid);
        delete test_ind_scaled
        system('svm-scale -r range test_ind.txt >> test_ind_scaled');
        system('svm-predict  -b 1  test_ind_scaled model output.txt>dump');
        load output.txt;
        Score(splits(iSplit).test,1) = output;
        SRCC(iSplit) = corr(Score(splits(iSplit).test),GT(splits(iSplit).test),'type','Spearman');

        [delta,beta,yhat,y,diff] = findrmse2(Score(splits(iSplit).test),GT(splits(iSplit).test));
        RMSE(iSplit) = sqrt(sum(diff.^2)/length(diff));
        PLCC(iSplit) = corr(GT(splits(iSplit).test), yhat, 'type','Pearson');
        KRCC(iSplit) = corr(GT(splits(iSplit).test), yhat, 'type','Kendall');
        PWRC(iSplit) = calPWRC(Score(splits(iSplit).test), MOS(splits(iSplit).test), SD(splits(iSplit).test));
    end
    
    SRCCall(cnt,:) = SRCC;
    RMSEall(cnt,:) = RMSE;
    PLCCall(cnt,:) = PLCC;
    KRCCall(cnt,:) = KRCC;
    PWRCall(cnt,:) = PWRC;
end
for cnt = 1:size(SRCCall,1)
    SRCCmean(cnt,1) = mean(SRCCall(cnt,:));
    SRCCmedian(cnt,1) = median(SRCCall(cnt,:));
    RMSEmean(cnt,1) = mean(RMSEall(cnt,:));
    RMSEmedian(cnt,1) = median(RMSEall(cnt,:));
    PLCCmean(cnt,1) = mean(PLCCall(cnt,:));
    PLCCmedian(cnt,1) = median(PLCCall(cnt,:));
    KRCCmean(cnt,1) = mean(KRCCall(cnt,:));
    KRCCmedian(cnt,1) = median(KRCCall(cnt,:));
    PWRCmean(cnt,1) = mean(PWRCall(cnt,:));
    PWRCmedian(cnt,1) = median(PWRCall(cnt,:));
end
cd ..
perSRCC_FR(8,:) = SRCCmean;
perPLCC_FR(8,:) = PLCCmean;
perKRCC_FR(8,:) = KRCCmean;
perRMSE_FR(8,:) = RMSEmean;
perPWRC_FR(8,:) = PWRCmean;

%% final results

% Type I
perSRCC_FR_TypeI = perSRCC_FR(1,:);
perPLCC_FR_TypeI = perPLCC_FR(1,:);
perKRCC_FR_TypeI = perKRCC_FR(1,:);
perRMSE_FR_TypeI = perRMSE_FR(1,:);
perPWRC_FR_TypeI = perPWRC_FR(1,:);

% Type II
perSRCC_FR_TypeII = perSRCC_FR(2,:);
perPLCC_FR_TypeII = perPLCC_FR(2,:);
perKRCC_FR_TypeII = perKRCC_FR(2,:);
perRMSE_FR_TypeII = perRMSE_FR(2,:);
perPWRC_FR_TypeII = perPWRC_FR(2,:);

% Type III
perSRCC_FR_TypeIII = perSRCC_FR(8,:);
perPLCC_FR_TypeIII = perPLCC_FR(8,:);
perKRCC_FR_TypeIII = perKRCC_FR(8,:);
perRMSE_FR_TypeIII = perRMSE_FR(8,:);
perPWRC_FR_TypeIII = perPWRC_FR(8,:);