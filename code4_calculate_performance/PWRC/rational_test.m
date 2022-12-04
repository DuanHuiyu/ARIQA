clc;clear;
close all;
path = '.\DB_image';
% mos & mos_std in ascending order
mos_std = [4.44;6.21;8.33;9.56;7.11];
mos = [5.15;33.34;46.62;50.12;62.45];
gt_rank = [5 4 3 2 1]; % gt_rank:   ground truth rank in descending order;
pred_rank = [5 1 3 2 4;% pred_rank: predicted rank by different IQA models
    2 4 3 5 1];
index = {'S8','S9'};   % Index:     please refer to Table I 
%% SRCC/KRCC/PWRC comparison between S8 and S9
metric = {'SRCC','KRCC','PWRC'};
RCI = zeros(3,2);
RCI(1,1) = corr(mos(pred_rank(1,:)),mos(gt_rank),'type','spearman');
RCI(1,2) = corr(mos(pred_rank(2,:)),mos(gt_rank),'type','spearman');
RCI(2,1) = corr(mos(pred_rank(1,:)),mos(gt_rank),'type','kendall');
RCI(2,2) = corr(mos(pred_rank(2,:)),mos(gt_rank),'type','kendall');
p = opinion_norm(mos,mos_std);
p.flag = 1;     % DMOS -> flag 0; MOS -> flag 1;
p.act  = 0;     % enable/disable A(x,T): p.act->1/0 
th = 0:0.5:110; % customize observation interval
[~,RCI(3,1)] = PWRC(mos(pred_rank(1,:)),mos(gt_rank),th,p); 
[~,RCI(3,2)] = PWRC(mos(pred_rank(2,:)),mos(gt_rank),th,p); 

for i = 1:3
    if RCI(i,1)>RCI(i,2)
        vs = 'superior to';
    elseif RCI(i,1)==RCI(i,2)
        vs = 'equivalent to';
    else
        vs = 'inferior to';
    end
    disp(['S8 is ',vs,' S9 in terms of ',metric{i}]);
end
%% Visual comparison between S8 and S9
TopN = numel(mos);
overlap = zeros(TopN,1);
for i=1:TopN
    GP1 = [mos_std(1),mos(1)];
    GP2 = [mos_std(i),mos(i)];
    overlap(i) = GP_overlap(GP1,GP2); 
end
% weight assigned to each image
w = overlap./sum(overlap);
figure;
for iter = 1:size(pred_rank,1)
    curr_rank = pred_rank(iter,:); 
    filter_result = 0;
    for i=1:TopN
        I = imread(fullfile(path,['fig_lena_rank',...
            num2str(curr_rank(i)),'.bmp']));
        % image fusion based on the predited rank
        filter_result = filter_result + w(i)*I;
    end
    s(iter) = subplot(1,size(pred_rank,1),iter);
    imshow(uint8(filter_result));
    title(['Fusion result of ',index{iter}]);
    imwrite(uint8(filter_result),...
        fullfile(path,['DB_result_',index{iter},'.bmp']));
end