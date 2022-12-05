
function [mse,psnr] = psnr_index(img1,img2)
MAX_PSNR = 1000;
mse = mean((img1(:)-img2(:)).^2);
psnr = 10*log10(255^2/mse);
psnr = min(MAX_PSNR, psnr);
