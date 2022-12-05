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
Path_Img_Ref1 = '..\database1_cfiqa\A\';
Path_Img_Ref2 = '..\database1_cfiqa\B\';
Path_Img_Dis = '..\database1_cfiqa\M\';
Path_Img_Ref1_sal = '..\database1_cfiqa\A_saliency\';
Path_Img_Ref2_sal = '..\database1_cfiqa\B_saliency\';
Path_Img_Dis_sal = '..\database1_cfiqa\M_saliency\';

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

for i = 1:30
    img_name{i,1} = [num2str(i,'%06.f'),'.png'];
end

%% run all traditional FR-IQA models
for cnt = 1:size(img_name,1)
    clc
    cnt
    
    img_reference1 = imread([Path_Img_Ref1,img_name{cnt,1}]);
    img1_1 = double(rgb2gray(img_reference1));
    imgA_1 = double(img_reference1);
    
    img_reference2 = imread([Path_Img_Ref2,img_name{cnt,1}]);
    img1_2 = double(rgb2gray(img_reference2));
    imgA_2 = double(img_reference2);
    
    img_distorted = imread([Path_Img_Dis,img_name{cnt,1}]);
    img2 = double(rgb2gray(img_distorted));
    imgB = double(img_distorted);
    
    img_reference1_sal = imread([Path_Img_Ref1_sal,img_name{cnt,1}]);
    img1_1_sal = double(img_reference1_sal);
    img_reference2_sal = imread([Path_Img_Ref2_sal,img_name{cnt,1}]);
    img1_2_sal = double(img_reference2_sal);
    img_distorted_sal = imread([Path_Img_Dis_sal,img_name{cnt,1}]);
    img2_sal = double(img_distorted_sal);


    %% FR 1
    [confusion_fsim_sal(cnt,1),confusion_fsimc_sal(cnt,1)] = FeatureSIM_sal(img1_1,img2,img1_1_sal);
    confusion_msssim_sal(cnt,1) = ms_ssim_index_sal(img1_1,img2,img1_1_sal);
    confusion_ssim1_sal(cnt,1) = ssim_index1_sal(img1_1,img2,img1_1_sal);
    confusion_ssim2_sal(cnt,1) = ssim_index2_sal(img1_1,img2,img1_1_sal);
    confusion_gmsm_sal(cnt,1) = GMSM_sal(img1_1,img2,img1_1_sal);

    [confusion_mse(cnt,1),confusion_psnr(cnt,1)] = psnr_index(img1_1,img2);
    viewing_angle = 1/3.5 * 180 / pi;
    dim = sqrt(prod(size(img1_1)));
    confusion_nqm(cnt,1) = nqm_modified(img1_1,img2,viewing_angle,dim);
    confusion_ssim1(cnt,1) = ssim_index1(img1_1,img2);
    confusion_ssim2(cnt,1) = ssim_index2(img1_1,img2);
    confusion_msssim(cnt,1) = ms_ssim_index(img1_1,img2);
    confusion_vif(cnt,1) = vifvec(img1_1,img2);
    confusion_vifp(cnt,1) = vifp_mscale(img1_1,img2);
    [confusion_iwssim(cnt,1),confusion_iwmse(cnt,1),confusion_iwpsnr(cnt,1)] = iwssim(img1_1,img2);
    [confusion_fsim(cnt,1),confusion_fsimc(cnt,1)] = FeatureSIM(imgA_1,imgB);
    confusion_gsi(cnt,1) = GSM(img1_1,img2);
    confusion_scssim(cnt,1) = scssim_index(img1_1,img2);
    confusion_gmsm(cnt,1) = GMSM(img1_1,img2);
    confusion_gmsd(cnt,1) = GMSD(img1_1,img2);
    confusion_pamse(cnt,1) = PAMSE(img1_1,img2);
    confusion_ltg(cnt,1) = ltg_index(imgA_1,imgB);
    confusion_vsi(cnt,1) = VSI(imgA_1,imgB);
    confusion_ifc(cnt,1) = ifcvec(img1_1,img2);

    
    %% FR 2
    [confusion_fsim_sal(cnt,2),confusion_fsimc_sal(cnt,2)] = FeatureSIM_sal(img1_2,img2,img1_2_sal);
    confusion_msssim_sal(cnt,2) = ms_ssim_index_sal(img1_2,img2,img1_2_sal);
    confusion_ssim1_sal(cnt,2) = ssim_index1_sal(img1_2,img2,img1_2_sal);
    confusion_ssim2_sal(cnt,2) = ssim_index2_sal(img1_2,img2,img1_2_sal);
    confusion_gmsm_sal(cnt,2) = GMSM_sal(img1_2,img2,img1_2_sal);

    [confusion_mse(cnt,2),confusion_psnr(cnt,2)] = psnr_index(img1_2,img2);
    viewing_angle = 1/3.5 * 180 / pi;
    dim = sqrt(prod(size(img1_2)));
    confusion_nqm(cnt,2) = nqm_modified(img1_2,img2,viewing_angle,dim);
    confusion_ssim1(cnt,2) = ssim_index1(img1_2,img2);
    confusion_ssim2(cnt,2) = ssim_index2(img1_2,img2);
    confusion_msssim(cnt,2) = ms_ssim_index(img1_2,img2);
    confusion_vif(cnt,2) = vifvec(img1_2,img2);
    confusion_vifp(cnt,2) = vifp_mscale(img1_2,img2);
    [confusion_iwssim(cnt,2),confusion_iwmse(cnt,2),confusion_iwpsnr(cnt,2)] = iwssim(img1_2,img2);
    [confusion_fsim(cnt,2),confusion_fsimc(cnt,2)] = FeatureSIM(imgA_2,imgB);
    confusion_gsi(cnt,2) = GSM(img1_2,img2);
    confusion_scssim(cnt,2) = scssim_index(img1_2,img2);
    confusion_gmsm(cnt,2) = GMSM(img1_2,img2);
    confusion_gmsd(cnt,2) = GMSD(img1_2,img2);
    confusion_pamse(cnt,2) = PAMSE(img1_2,img2);
    confusion_ltg(cnt,2) = ltg_index(imgA_2,imgB);
    confusion_vsi(cnt,2) = VSI(imgA_2,imgB);
    confusion_ifc(cnt,2) = ifcvec(img1_2,img2);
end
