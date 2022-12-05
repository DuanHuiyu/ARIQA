function swn = snw_ssim_index(img1,img2)
%==========================================================================
% 1) Please cite the paper (K. Gu, G. Zhai, X. Yang, L. Chen, and W. Zhang,
% "Nonlinear additive model based saliency map weighting strategy for image
% quality assessment," in Proc. IEEE Int. Workshop on Multimedia Signal
% Process., pp. 313-318, Sept. 2012.)
% 2) If any question, please contact me through guke.doctor@gmail.com; 
% gukesjtuee@gmail.com. 
% 3) Welcome to cooperation, and I am very willing to share my experience.
%==========================================================================
path = pwd;
cd E:\IQA_algorithm\FRIQA\SNW
addpath 'SaliencyToolbox2.2\img'
addpath 'SaliencyToolbox2.2\img2'
addpath 'SaliencyToolbox2.2\src'
addpath 'SaliencyToolbox2.2\mex'
addpath 'SaliencyToolbox2.2\mfiles'

% img1 = imread(img1_name);
% img2 = imread(img2_name);
runSaliency(img2);
load sal
sal3 = sal1;
runSaliency(img1);
load sal
img1 = rgb2gray(img1);
img2 = rgb2gray(img2);
img1 = double(img1);
img2 = double(img2);

window = fspecial('gaussian', 11, 1.5);
K(1) = 0.01;
K(2) = 0.03;
L = 255;
C1 = (K(1)*L)^2;
C2 = (K(2)*L)^2;
window = window/sum(sum(window));

[m,n,l] = size(img1);
f = max(1,round(min(m,n)/256));
if(f>1)
lpf = ones(f,f);
lpf = lpf/sum(lpf(:));
img1 = imfilter(img1,lpf,'symmetric','same');
img2 = imfilter(img2,lpf,'symmetric','same');
img1 = img1(1:f:end,1:f:end);
img2 = img2(1:f:end,1:f:end);
sal1 = imfilter(sal1,lpf,'symmetric','same');
sal3 = imfilter(sal3,lpf,'symmetric','same');
sal1 = sal1(1:f:end,1:f:end);
sal3 = sal3(1:f:end,1:f:end);
end

mu1    = filter2(window, img1, 'valid');
mu2    = filter2(window, img2, 'valid');
mu1_sq = mu1.*mu1;
mu2_sq = mu2.*mu2;
mu1_mu2 = mu1.*mu2;
sigma1_sq = filter2(window, img1.*img1, 'valid') - mu1_sq;
sigma2_sq = filter2(window, img2.*img2, 'valid') - mu2_sq;
sigma12  = filter2(window, img1.*img2, 'valid') - mu1_mu2;
ssim_map = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))./((mu1_sq + mu2_sq + C1).*(sigma1_sq + sigma2_sq + C2));

xxx = sal1(6:end-5,6:end-5);
yyy = sal3(6:end-5,6:end-5);
delete sal.mat
map = xxx+yyy-1.7*min(xxx,yyy);
swn = mean2(ssim_map.*map)/mean2(map);
cd(path)
