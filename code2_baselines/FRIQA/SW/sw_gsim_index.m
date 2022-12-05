function swm = sw_gsim_index(img1,img2)
%==========================================================================
% 1) Please cite the paper (K. Gu, G. Zhai, X. Yang, W. Zhang, and M. Liu, 
% "Structural similarity weighting for image quality assessment," in Proc. 
% IEEE Int. Conf. Multimedia and Expo Workshops, pp. 1-6, Jul. 2013.)
% 2) If any question, please contact me through guke.doctor@gmail.com; 
% gukesjtuee@gmail.com. 
% 3) Welcome to cooperation, and I am very willing to share my experience.
%==========================================================================
[M N] = size(img1);
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
%%
c = [1 1 1 1 0.25 0.25 0.25 0.25];
l = 4;
m = 1;
n = 7;
swm = swgsim_saliencyz(img1,img2,l,n,c,m,gsim_map);
%%
function mgsim = swgsim_saliencyz(img1,img2,l,n,c,m,xxx_map)
tmp = zeros(8,1);
[M N] = size(img1);
mgsim1 = zeros(length(l),length(n),size(c,1),length(m));
mgsim2 = zeros(length(l),length(n),size(c,1),length(m));
%%
[M N] = size(img1);
[zzz,gsim_map] = swgsim_index(img1,img2);
gsim_map = xxx_map;
%%
for i = 1:length(l)
for j = 1:length(n)
sal1 = zeros(M,N,size(c,1));
for k = 1:length(m)
for s = 1+l(i)+floor(n(j)/2):l(i)/m(k):M-floor(n(j)/2)-l(i)-l(i)+1
for t = 1+l(i)+floor(n(j)/2):l(i)/m(k):N-floor(n(j)/2)-l(i)-l(i)+1
blk = img1(s-floor(n(j)/2)-l(i)+1:s+l(i)-1+floor(n(j)/2)-l(i)+1,t-floor(n(j)/2)-l(i)+1:t+l(i)-1+floor(n(j)/2)-l(i)+1);
blk2 = img1(s-floor(n(j)/2)-l(i):s+l(i)-1+floor(n(j)/2)-l(i),t-floor(n(j)/2):t+l(i)-1+floor(n(j)/2));
tmp(1) = swgsim_index(blk,blk2);%ио
blk2 = img1(s-floor(n(j)/2):s+l(i)-1+floor(n(j)/2),t-floor(n(j)/2)+l(i):t+l(i)-1+floor(n(j)/2)+l(i));
tmp(2) = swgsim_index(blk,blk2);%ср
blk2 = img1(s-floor(n(j)/2)+l(i):s+l(i)-1+floor(n(j)/2)+l(i),t-floor(n(j)/2):t+l(i)-1+floor(n(j)/2));
tmp(3) = swgsim_index(blk,blk2);%об
blk2 = img1(s-floor(n(j)/2):s+l(i)-1+floor(n(j)/2),t-floor(n(j)/2)-l(i):t+l(i)-1+floor(n(j)/2)-l(i));
tmp(4) = swgsim_index(blk,blk2);%вС
blk2 = img1(s-floor(n(j)/2)-l(i):s+l(i)-1+floor(n(j)/2)-l(i),t-floor(n(j)/2)+l(i):t+l(i)-1+floor(n(j)/2)+l(i));
tmp(5) = swgsim_index(blk,blk2);%1
blk2 = img1(s-floor(n(j)/2)+l(i):s+l(i)-1+floor(n(j)/2)+l(i),t-floor(n(j)/2)+l(i):t+l(i)-1+floor(n(j)/2)+l(i));
tmp(6) = swgsim_index(blk,blk2);%2
blk2 = img1(s-floor(n(j)/2)+l(i):s+l(i)-1+floor(n(j)/2)+l(i),t-floor(n(j)/2)-l(i):t+l(i)-1+floor(n(j)/2)-l(i));
tmp(7) = swgsim_index(blk,blk2);%3
blk2 = img1(s-floor(n(j)/2)-l(i):s+l(i)-1+floor(n(j)/2)-l(i),t-floor(n(j)/2)-l(i):t+l(i)-1+floor(n(j)/2)-l(i));
tmp(8) = swgsim_index(blk,blk2);%4
for q = 1:size(c,1)
sal1(s:s+l(i)-1,t:t+l(i)-1,q) = 1-c(q,:)*tmp/sum(c(q,:));
end
end;
end;
for q = 1:size(c,1)
mgsim1(i,j,q,k) = sum(sum(gsim_map.*sal1(:,:,q)))/sum(sum(sal1(:,:,q)));
end
end
end;
end;
mgsim = mgsim1;
%%
function [mgsim,gsim_map] = swgsim_index(img1,img2)
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
mgsim = mean2(gsim_map);