clc
clear

load ..\DEHAZEfrTest\Results\MOS.mat
load Feature\DEHAZEfrScoreFR.mat
load Feature\dehaze_fr.mat

index = 1:480;

score(:,1) = dehaze_psnr(index);
score(:,2) = dehaze_ssim1(index);
score(:,3) = dehaze_msssim(index);
score(:,4) = dehaze_vif(index);
score(:,5) = dehaze_mad(index);
score(:,6) = dehaze_iwssim(index);
score(:,7) = dehaze_gsi(index);
score(:,8) = dehaze_fsim(index);
score(:,9) = dehaze_gmsd(index);
score(:,10) = dehaze_psim(index);
score(:,11) = dehaze_fr(index,1);
GT = MOSz(index);

addpath('E:\DATABASE\fit\')
%% performance
for i = 1:size(score,2)
    perSRCC(i) = corr(score(:,i),GT,'type','Spearman');
    [delta,beta,yhat,y,diff] = findrmse2(score(:,i),GT);
    score_mapped(:,i) = yhat;
    perRMSE(i) = sqrt(sum(diff.^2)/length(diff));
    perPLCC(i) = corr(GT, yhat, 'type','Pearson');
    perPLCCo(i) = corr(score(:,i),GT,'type','Pearson');
    failNum = 1;
    while (abs(perPLCC(i)) - abs(perPLCCo(i)) < 0.0001) && (failNum<10)
        failNum = failNum+1
        [delta,beta,yhat,y,diff] = findrmse2_p(score(:,i),GT,failNum);
        perRMSE(i) = sqrt(sum(diff.^2)/length(diff));
        perPLCC(i) = corr(GT, yhat, 'type','Pearson');
    end
end

%% scatter plot
% DistortionType = repmat([1:8]',[45 1]);
% figure
% hold on
% alg = 11;
% index = find(DistortionType==1);
% scatter(score(index,alg),GT(index),35,'ro','filled')
% index = find(DistortionType==2);
% scatter(score(index,alg),GT(index),35,'g^','filled')
% index = find(DistortionType==3);
% scatter(score(index,alg),GT(index),35,'bv','filled')
% index = find(DistortionType==4);
% scatter(score(index,alg),GT(index),35,'y<','filled')
% index = find(DistortionType==5);
% scatter(score(index,alg),GT(index),35,'m>','filled')
% index = find(DistortionType==6);
% scatter(score(index,alg),GT(index),35,'cs','filled')
% index = find(DistortionType==7);
% scatter(score(index,alg),GT(index),35,[0.8 0.8 0.8],'d','filled')
% index = find(DistortionType==8);
% scatter(score(index,alg),GT(index),35,'kp','filled')
% addpath('fit\');
% [delta,beta,yhat,y,diff] = findrmse2(score(:,alg),GT);
% x_min = min(score(:,alg));
% x_max = max(score(:,alg));
% x = x_min:(x_max-x_min)/200:x_max;
% F = beta(1)*(0.5-1./(1+exp(beta(2)*(x-beta(3)))))+beta(4)*x+beta(5);
% plot(x,F,'k','LineWidth',1);
% ylim([0 85])
% xlim([x_min-(x_max-x_min)*0.03 x_max+(x_max-x_min)*0.03])
% set(gca,'fontname','Times New Roman','Fontsize',16)
% h = legend('Berman16','Cai16','Fattal08','He09','Lai15','Meng13','Tarel09','Xiao12','Location','northwest');%southeast
% set(h,'Fontsize',12);
% xlabel('Proposed','fontname','Times New Roman','fontsize',20);  
% ylabel('MOS','fontname','Times New Roman','fontsize',20);  
% box on

%% Statistical Significance Test
residual = score_mapped - repmat(GT,[1 size(score_mapped,2)]);
for i = 1:size(residual,2)
    for j = 1:size(residual,2)
        hh_greater(i,j) = vartest2(residual(:,i),residual(:,j),'Tail','right');
        hh_less(i,j) = vartest2(residual(:,i),residual(:,j),'Tail','left');
    end
end
hh = hh_less - hh_greater;

All = ones(size(hh_greater))*0.5;
index = find(hh_less==1);
All(index) = 1;
index = find(hh_greater==1);
All(index) = 0;

fontsize = 13;
figure
All(12,:) = nan;
All(:,12) = nan;
pcolor(All)
colormap(gray(3))
axis ij
axis square
set(gca,'XTick',1.5:1:14);
set(gca,'XTicklabel',{'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I','J', 'K'})
set(gca,'YTick',1.5:1:14);
set(gca,'YTicklabel',{'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I','J', 'K'})
set(gca,'XAxisLocation','top')
set(gca,'fontname','Times New Roman','fontsize',fontsize);

