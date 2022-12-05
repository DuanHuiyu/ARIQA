% ------------------------------------------------------------------------------------------
% Confusing image quality assessment: Towards better augmented reality experience
% Huiyu Duan, Xiongkuo Min, Yucheng Zhu, Guangtao Zhai, Xiaokang Yang, and Patrick Le Callet
% IEEE Transactions on Image Processing (TIP)
% ------------------------------------------------------------------------------------------

clc
close all
clear all
warning off

Path = '.\';
Path_Img_Dis = '..\database2_ariqa\img_Mixed\cropped\';
Path_Img_AR = '..\database2_ariqa\img_AR\reference\';
Path_Img_BG = '..\database2_ariqa\img_BG\cropped\';

Path_Img_AR2 = '..\database2_ariqa\img_AR\distorted\';

addpath([Path 'TOOL\matlabPyrTools'])
addpath([Path 'TOOL\dwt2d'])
addpath([Path 'TOOL\PhaseCongruency'])
addpath([Path 'TOOL\libsvm-3.12\matlab'])
addpath([Path 'TOOL\libsvm-3.17\matlab'])
addpath([Path 'FRIQA\PSNR'])
addpath([Path 'FRIQA\NQM'])
addpath([Path 'FRIQA\SSIM'])
addpath([Path 'FRIQA\MS'])
addpath([Path 'FRIQA\IFC'])
addpath([Path 'FRIQA\VIF'])
addpath([Path 'FRIQA\VIFP'])
addpath([Path 'FRIQA\VSNR'])
addpath([Path 'FRIQA\MAD'])
addpath([Path 'FRIQA\IW'])
addpath([Path 'FRIQA\FSIM'])
addpath([Path 'FRIQA\GSI'])
addpath([Path 'FRIQA\IGM'])
addpath([Path 'FRIQA\SR'])
addpath([Path 'FRIQA\SC'])
addpath([Path 'FRIQA\SNW'])
addpath([Path 'FRIQA\SW'])
addpath([Path 'FRIQA\PAMSE'])
addpath([Path 'FRIQA\GMSD'])
addpath([Path 'FRIQA\LTG'])
addpath([Path 'FRIQA\VSI'])


for i = 1:560
    img_name_dis{i,1} = [num2str(i),'.png'];
end
for i = 1:560
    cnt = ceil((mod((i-1),140)+1)/7);
    img_name_ref{i,1} = [num2str(cnt),'.png'];
end

for cnt = 1:140
    i = ceil(cnt/7);
    j = mod((cnt-1),7);
    if ((j==1) || (j==2))
        img_name_dis2{cnt,1} = [num2str(i),'_',num2str(j),'.jpg'];
    else
        img_name_dis2{cnt,1} = [num2str(i),'_',num2str(j),'.png'];
    end
end

alpha = [0.26, 0.42, 0.58, 0.74];
%% run all traditional FR-IQA models
for cnt = 1:size(img_name_dis,1)
    clc
    cnt
    
    img_reference = imread([Path_Img_AR,img_name_ref{cnt,1}]);
    img1_a = double(rgb2gray(img_reference));
    imgA_a = double(img_reference);
    
    img_distorted = imread([Path_Img_Dis,img_name_dis{cnt,1}]);
    img2 = double(rgb2gray(img_distorted));
    imgB = double(img_distorted);
    
    img_background = imread([Path_Img_BG,img_name_ref{cnt,1}]);
    img1_b = double(rgb2gray(img_background));
    imgA_b = double(img_background);
    
    i = ceil(cnt/140);
    j = mod((cnt-1),140);
    img_distorted_AR = imread([Path_Img_AR2,img_name_dis2{j+1,1}])*alpha(i);
    img2_AR = double(rgb2gray(img_distorted_AR));
    imgB_AR = double(img_distorted_AR);

    %% FR type I
    [confusion_mse(cnt,1),confusion_psnr(cnt,1)] = psnr_index(img1_a,img2_AR);
    viewing_angle = 1/3.5 * 180 / pi;
    dim = sqrt(prod(size(img1_a)));
    confusion_nqm(cnt,1) = nqm_modified(img1_a,img2_AR,viewing_angle,dim);
    confusion_ssim1(cnt,1) = ssim_index1(img1_a,img2_AR);
    confusion_ssim2(cnt,1) = ssim_index2(img1_a,img2_AR);
    confusion_msssim(cnt,1) = ms_ssim_index(img1_a,img2_AR);
    confusion_vif(cnt,1) = vifvec(img1_a,img2_AR);
    confusion_vifp(cnt,1) = vifp_mscale(img1_a,img2_AR);
    [confusion_iwssim(cnt,1),confusion_iwmse(cnt,1),confusion_iwpsnr(cnt,1)] = iwssim(img1_a,img2_AR);
    [confusion_fsim(cnt,1),confusion_fsimc(cnt,1)] = FeatureSIM(imgA_a,imgB_AR);
    confusion_gsi(cnt,1) = GSM(img1_a,img2_AR);
    confusion_scssim(cnt,1) = scssim_index(img1_a,img2_AR);
    confusion_gmsm(cnt,1) = GMSM(img1_a,img2_AR);
    confusion_gmsd(cnt,1) = GMSD(img1_a,img2_AR);
    confusion_pamse(cnt,1) = PAMSE(img1_a,img2_AR);
    confusion_ltg(cnt,1) = ltg_index(imgA_a,imgB_AR);
    confusion_vsi(cnt,1) = VSI(imgA_a,imgB_AR);
    confusion_ifc(cnt,1) = ifcvec(img1_a,img2_AR);

    %% FR a
    [confusion_mse_a(cnt,1),confusion_psnr_a(cnt,1)] = psnr_index(img1_a,img2);
    viewing_angle = 1/3.5 * 180 / pi;
    dim = sqrt(prod(size(img1_a)));
    confusion_nqm_a(cnt,1) = nqm_modified(img1_a,img2,viewing_angle,dim);
    confusion_ssim1_a(cnt,1) = ssim_index1(img1_a,img2);
    confusion_ssim2_a(cnt,1) = ssim_index2(img1_a,img2);
    confusion_msssim_a(cnt,1) = ms_ssim_index(img1_a,img2);
    confusion_vif_a(cnt,1) = vifvec(img1_a,img2);
    confusion_vifp_a(cnt,1) = vifp_mscale(img1_a,img2);
    [confusion_iwssim_a(cnt,1),confusion_iwmse_a(cnt,1),confusion_iwpsnr_a(cnt,1)] = iwssim(img1_a,img2);
    [confusion_fsim_a(cnt,1),confusion_fsimc_a(cnt,1)] = FeatureSIM(imgA_a,imgB);
    confusion_gsi_a(cnt,1) = GSM(img1_a,img2);
    confusion_scssim_a(cnt,1) = scssim_index(img1_a,img2);
    confusion_gmsm_a(cnt,1) = GMSM(img1_a,img2);
    confusion_gmsd_a(cnt,1) = GMSD(img1_a,img2);
    confusion_pamse_a(cnt,1) = PAMSE(img1_a,img2);
    confusion_ltg_a(cnt,1) = ltg_index(imgA_a,imgB);
    confusion_vsi_a(cnt,1) = VSI(imgA_a,imgB);
    confusion_ifc_a(cnt,1) = ifcvec(img1_a,img2);
    
    %% FR b
    [confusion_mse_b(cnt,1),confusion_psnr_b(cnt,1)] = psnr_index(img1_b,img2);
    viewing_angle = 1/3.5 * 180 / pi;
    dim = sqrt(prod(size(img1_b)));
    confusion_nqm_b(cnt,1) = nqm_modified(img1_b,img2,viewing_angle,dim);
    confusion_ssim1_b(cnt,1) = ssim_index1(img1_b,img2);
    confusion_ssim2_b(cnt,1) = ssim_index2(img1_b,img2);
    confusion_msssim_b(cnt,1) = ms_ssim_index(img1_b,img2);
    confusion_vif_b(cnt,1) = vifvec(img1_b,img2);
    confusion_vifp_b(cnt,1) = vifp_mscale(img1_b,img2);
    [confusion_iwssim_b(cnt,1),confusion_iwmse_b(cnt,1),confusion_iwpsnr_b(cnt,1)] = iwssim(img1_b,img2);
    [confusion_fsim_b(cnt,1),confusion_fsimc_b(cnt,1)] = FeatureSIM(imgA_b,imgB);
    confusion_gsi_b(cnt,1) = GSM(img1_b,img2);
    confusion_scssim_b(cnt,1) = scssim_index(img1_b,img2);
    confusion_gmsm_b(cnt,1) = GMSM(img1_b,img2);
    confusion_gmsd_b(cnt,1) = GMSD(img1_b,img2);
    confusion_pamse_b(cnt,1) = PAMSE(img1_b,img2);
    confusion_ltg_b(cnt,1) = ltg_index(imgA_b,imgB);
    confusion_vsi_b(cnt,1) = VSI(imgA_b,imgB);
    confusion_ifc_b(cnt,1) = ifcvec(img1_b,img2);
end