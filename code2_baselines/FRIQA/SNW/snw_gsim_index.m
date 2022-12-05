function swn = snw_gsim_index(img1,img2)
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

dx = [3 0 -3;10 0 -10;3  0  -3]/16;
dy = dx';
xx1 = conv2(img1, dx, 'same');     
xy1 = conv2(img1, dy, 'same');    
gradientMap1 = sqrt(xx1.^2 + xy1.^2);
xx2 = conv2(img2, dx, 'same');     
xy2 = conv2(img2, dy, 'same');    
gradientMap2 = sqrt(xx2.^2 + xy2.^2);
G1 = 2*gradientMap1.*gradientMap2;
G2 = gradientMap1.^2 + gradientMap2.^2;
gsim_map = (G1 + 170) ./(G2 + 170);

xxx = sal1;
yyy = sal3;
delete sal.mat
map = xxx+yyy-1.7*min(xxx,yyy);
swn = mean2(gsim_map.*map)/mean2(map);
cd(path)
