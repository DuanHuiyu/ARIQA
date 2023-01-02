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

results_path = {'..\code3_cfiqa_ariqa\results\checkpoints\ariqa\test.csv',
    '..\code3_cfiqa_ariqa\results\checkpoints\ariqa_plus\test.csv',};

gt_path = '..\code3_cfiqa_ariqa\train_test_split\ARIQA\';
cross_num = 5;

MOS = load('..\database2_ariqa\MOS.mat').MOS;
SD = load('..\database2_ariqa\SD.mat').SD;
MOSz = load('..\database2_ariqa\MOSz.mat').MOSz;
GT = MOSz;

%%
for cnt = 0:(cross_num-1)
    [numeric_r,text_r,raw_r] = xlsread([gt_path,'ARIQA_test_',num2str(cnt),'.csv']);
    GT = numeric_r(1:280);
    
    % find corresponding MOSz value
    text = text_r(:,3);
    for i = 1:size(text,1)
        temp = strsplit(text{i},{'/','.'});
        Index(i) = str2num(temp{end-1});
    end
    
    GTz = MOSz(Index);
    MOSNew = MOS(Index);
    SDNew = SD(Index);
    
    for i = 1:size(results_path,1)
        file_name = results_path{i,1};
        [numeric_r,text_r,raw_r] = xlsread([file_name,'test',num2str(cnt),'_best.csv']);
        results = numeric_r(1:280);

        perSRCC_FR(cnt+1,i) = corr(results,GTz,'type','Spearman');
        perKRCC_FR(cnt+1,i) = corr(results,GTz,'type','Kendall');
        [delta,beta,yhat,y,diff] = findrmse2(results,GTz);
        score_mapped(:,i) = yhat;
        perRMSE_FR(cnt+1,i) = sqrt(sum(diff.^2)/length(diff));
        perPLCC_FR(cnt+1,i) = corr(GTz, yhat, 'type','Pearson');
        perPWRC_FR(cnt+1,i) = calPWRC(results, MOSNew, SDNew);
    end
end

meanSRCC = mean(perSRCC_FR(:,:),1);
meanKRCC = mean(perKRCC_FR(:,:),1);
meanPLCC = mean(perPLCC_FR(:,:),1);
meanRMSE = mean(perRMSE_FR(:,:),1);
meanPWRC = mean(perPWRC_FR(:,:),1);
