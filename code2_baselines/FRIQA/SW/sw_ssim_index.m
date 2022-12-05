function swm = sw_ssim_index(img1,img2)
%==========================================================================
% 1) Please cite the paper (K. Gu, G. Zhai, X. Yang, W. Zhang, and M. Liu, 
% "Structural similarity weighting for image quality assessment," in Proc. 
% IEEE Int. Conf. Multimedia and Expo Workshops, pp. 1-6, Jul. 2013.)
% 2) If any question, please contact me through guke.doctor@gmail.com; 
% gukesjtuee@gmail.com. 
% 3) Welcome to cooperation, and I am very willing to share my experience.
%==========================================================================
[M N] = size(img1);
window = fspecial('gaussian', 11, 1.5);
K(1) = 0.01;
K(2) = 0.03;
L = 255;
C1 = (K(1)*L)^2;
C2 = (K(2)*L)^2;
window = window/sum(sum(window));
%%
f = max(1,round(min(M,N)/256));
if(f>1)
lpf = ones(f,f);
lpf = lpf/sum(lpf(:));
img1 = imfilter(img1,lpf,'symmetric','same');
img2 = imfilter(img2,lpf,'symmetric','same');
img1 = img1(1:f:end,1:f:end);
img2 = img2(1:f:end,1:f:end);
end
%%
mu1 = filter2(window, img1, 'valid');
mu2 = filter2(window, img2, 'valid');
mu1_sq = mu1.*mu1;
mu2_sq = mu2.*mu2;
mu1_mu2 = mu1.*mu2;
sigma1_sq = filter2(window, img1.*img1, 'valid') - mu1_sq;
sigma2_sq = filter2(window, img2.*img2, 'valid') - mu2_sq;
sigma12 = filter2(window, img1.*img2, 'valid') - mu1_mu2;
ssim_map = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))./((mu1_sq + mu2_sq + C1).*(sigma1_sq + sigma2_sq + C2));
%%
c = [1 1 1 1 0.25 0.25 0.25 0.25];
l = 4;
m = 1;
n = 7;
swm = swssim_saliencyz(img1,img2,l,n,c,m,ssim_map);
%%
function mssim = swssim_saliencyz(img1,img2,l,n,c,m,xxx_map)
tmp = zeros(8,1);
[M N] = size(img1);
mssim1 = zeros(length(l),length(n),size(c,1),length(m));
mssim2 = zeros(length(l),length(n),size(c,1),length(m));
%%
[M N] = size(img1);
[zzz,ssim_map] = swssim_index1(img1,img2,11);
ssim_map = xxx_map;
%%
for i = 1:length(l)
for j = 1:length(n)
sal1 = zeros(M,N,size(c,1));
for k = 1:length(m)
for s = 1+l(i)+floor(n(j)/2):l(i)/m(k):M-floor(n(j)/2)-l(i)-l(i)+1
for t = 1+l(i)+floor(n(j)/2):l(i)/m(k):N-floor(n(j)/2)-l(i)-l(i)+1
blk = img1(s-floor(n(j)/2)-l(i)+1:s+l(i)-1+floor(n(j)/2)-l(i)+1,t-floor(n(j)/2)-l(i)+1:t+l(i)-1+floor(n(j)/2)-l(i)+1);
blk2 = img1(s-floor(n(j)/2)-l(i):s+l(i)-1+floor(n(j)/2)-l(i),t-floor(n(j)/2):t+l(i)-1+floor(n(j)/2));
tmp(1) = swssim_index2(blk,blk2,n(j));%ио
blk2 = img1(s-floor(n(j)/2):s+l(i)-1+floor(n(j)/2),t-floor(n(j)/2)+l(i):t+l(i)-1+floor(n(j)/2)+l(i));
tmp(2) = swssim_index2(blk,blk2,n(j));%ср
blk2 = img1(s-floor(n(j)/2)+l(i):s+l(i)-1+floor(n(j)/2)+l(i),t-floor(n(j)/2):t+l(i)-1+floor(n(j)/2));
tmp(3) = swssim_index2(blk,blk2,n(j));%об
blk2 = img1(s-floor(n(j)/2):s+l(i)-1+floor(n(j)/2),t-floor(n(j)/2)-l(i):t+l(i)-1+floor(n(j)/2)-l(i));
tmp(4) = swssim_index2(blk,blk2,n(j));%вС
blk2 = img1(s-floor(n(j)/2)-l(i):s+l(i)-1+floor(n(j)/2)-l(i),t-floor(n(j)/2)+l(i):t+l(i)-1+floor(n(j)/2)+l(i));
tmp(5) = swssim_index2(blk,blk2,n(j));%1
blk2 = img1(s-floor(n(j)/2)+l(i):s+l(i)-1+floor(n(j)/2)+l(i),t-floor(n(j)/2)+l(i):t+l(i)-1+floor(n(j)/2)+l(i));
tmp(6) = swssim_index2(blk,blk2,n(j));%2
blk2 = img1(s-floor(n(j)/2)+l(i):s+l(i)-1+floor(n(j)/2)+l(i),t-floor(n(j)/2)-l(i):t+l(i)-1+floor(n(j)/2)-l(i));
tmp(7) = swssim_index2(blk,blk2,n(j));%3
blk2 = img1(s-floor(n(j)/2)-l(i):s+l(i)-1+floor(n(j)/2)-l(i),t-floor(n(j)/2)-l(i):t+l(i)-1+floor(n(j)/2)-l(i));
tmp(8) = swssim_index2(blk,blk2,n(j));%4
for q = 1:size(c,1)
sal1(s:s+l(i)-1,t:t+l(i)-1,q) = 1-c(q,:)*tmp/sum(c(q,:));
end
end;
end;
for q = 1:size(c,1)
mssim1(i,j,q,k) = sum(sum(ssim_map.*sal1(6:end-5,6:end-5,q)))/sum(sum(sal1(6:end-5,6:end-5,q)));
end
end
end;
end;
mssim = mssim1;
%%
function [mssim,ssim_map] = swssim_index1(img1,img2,n)
[M N] = size(img1);
window = fspecial('gaussian',n,1.5);	
K(1) = 0.01;
K(2) = 0.03;
L = 255;
C1 = (K(1)*L)^2;
C2 = (K(2)*L)^2;
mu1 = filter2(window, img1, 'valid');
mu2 = filter2(window, img2, 'valid');
mu1_sq = mu1.*mu1;
mu2_sq = mu2.*mu2;
mu1_mu2 = mu1.*mu2;
sigma1_sq = filter2(window, img1.*img1, 'valid') - mu1_sq;
sigma2_sq = filter2(window, img2.*img2, 'valid') - mu2_sq;
sigma12 = filter2(window, img1.*img2, 'valid') - mu1_mu2;
ssim_map = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))./((mu1_sq + mu2_sq + C1).*(sigma1_sq + sigma2_sq + C2));
mssim = mean(ssim_map(:));
%%
function [mssim,ssim_map] = swssim_index2(img1,img2,n)
[M N] = size(img1);
window = fspecial('gaussian',n,1.5);	
K(1) = 0.01;
K(2) = 0.03;
L = 255;
C1 = (K(1)*L)^2;
C2 = (K(2)*L)^2;
mu1 = filter2(window, img1, 'valid');
mu2 = filter2(window, img2, 'valid');
mu1_sq = mu1.*mu1;
mu2_sq = mu2.*mu2;
mu1_mu2 = mu1.*mu2;
sigma1_sq = filter2(window, img1.*img1, 'valid') - mu1_sq;
sigma2_sq = filter2(window, img2.*img2, 'valid') - mu2_sq;
sigma12 = filter2(window, img1.*img2, 'valid') - mu1_mu2;
ssim_map = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))./((mu1_sq + mu2_sq + C1).*(sigma1_sq + sigma2_sq + C2));
mssim = mean(ssim_map(:));