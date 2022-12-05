
function [mse_sal,psnr_sal,mse_sal2,psnr_sal2] = psnr_index_sal(img1,img2,sal,sal2)
MAX_PSNR = 1000;
% mse = mean((img1(:)-img2(:)).^2);
mse = (img1-img2).^2;
mse_sal = sum(sum(mse.*sal)) / sum(sum(sal));
mse_sal2 = sum(sum(mse.*sal2)) / sum(sum(sal2));

% psnr = 10*log10(255^2/mse);
% psnr = min(MAX_PSNR, psnr);
psnr = 10*log10(255^2./mse);
psnr = min(MAX_PSNR, psnr);
psnr_sal = sum(sum(psnr.*sal)) / sum(sum(sal));
psnr_sal2 = sum(sum(psnr.*sal2)) / sum(sum(sal2));

