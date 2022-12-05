function sc = scssim_index(img1,img2)
%==========================================================================
% 1) Please cite the paper (K. Gu, G. Zhai, X. Yang, and W. Zhang, "An
% improved full reference image quality metric based on structure 
% compensation", in APSIPA ASC, Dec. 2012.)
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
img1 = double(img1);
img2 = double(img2);
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
l_map = (2*mu1_mu2 + C1)./(mu1_sq + mu2_sq + C1);
c_map = real((2*sqrt(sigma1_sq).*sqrt(sigma2_sq)+ C2)./(sigma1_sq + sigma2_sq + C2));
s_map = real((2*sigma12 + C2)./(2*sqrt(sigma1_sq).*sqrt(sigma2_sq)+ C2));
lssim = mean2(l_map);
cssim = mean2(c_map);
sssim = mean2(s_map);
%%
imgX = img1(6:end-5,6:end-5);
imgY = img2(6:end-5,6:end-5);
%% reference(l)
rmu1 = filter2(window, imgX, 'valid');
rmu2 = filter2(window, mu1, 'valid');
rmu1_sq = rmu1.*rmu1;
rmu2_sq = rmu2.*rmu2;
rmu1_mu2 = rmu1.*rmu2;
rsigma1_sq = filter2(window, imgX.*imgX, 'valid') - rmu1_sq;
rsigma2_sq = filter2(window, mu1.*mu1, 'valid') - rmu2_sq;
rsigma12 = filter2(window, imgX.*mu1, 'valid') - rmu1_mu2;
rl_map = (2*rmu1_mu2 + C1)./(rmu1_sq + rmu2_sq + C1);
rc_map = real((2*sqrt(rsigma1_sq).*sqrt(rsigma2_sq)+ C2)./(rsigma1_sq + rsigma2_sq + C2));
rs_map = real((2*rsigma12 + C2)./(2*sqrt(rsigma1_sq).*sqrt(rsigma2_sq)+ C2));
rssim_map = rl_map.*rc_map.*rs_map;
rssim1 = mean2(rssim_map);
%% distorted(l)
dmu1 = filter2(window, imgY, 'valid');
dmu2 = filter2(window, mu2, 'valid');
dmu1_sq = dmu1.*dmu1;
dmu2_sq = dmu2.*dmu2;
dmu1_mu2 = dmu1.*dmu2;
dsigma1_sq = filter2(window, imgY.*imgY, 'valid') - dmu1_sq;
dsigma2_sq = filter2(window, mu2.*mu2, 'valid') - dmu2_sq;
dsigma12 = filter2(window, imgY.*mu2, 'valid') - dmu1_mu2;
dl_map = (2*dmu1_mu2 + C1)./(dmu1_sq + dmu2_sq + C1);
dc_map = real((2*sqrt(dsigma1_sq).*sqrt(dsigma2_sq)+ C2)./(dsigma1_sq + dsigma2_sq + C2));
ds_map = real((2*dsigma12 + C2)./(2*sqrt(dsigma1_sq).*sqrt(dsigma2_sq)+ C2));
dssim_map = dl_map.*dc_map.*ds_map;
dssim1 = mean2(dssim_map);
%%
lztmp = rssim1-dssim1;
lztmp2 = lztmp;
lztmp(lztmp>0) = -(lztmp(lztmp>0)).^0.65;
coff = [1.5043 2.1988 -2.8386 4.8668 0.9415 1.0484 0.6025];
sc = real(lssim^coff(5)*cssim^coff(6)*sssim^coff(7)+coff(1)*lztmp^coff(2)+coff(3)*lztmp2^coff(4));
