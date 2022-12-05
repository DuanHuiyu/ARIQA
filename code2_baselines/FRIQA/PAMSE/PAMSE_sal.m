function [score_sal,score_sal2] =  PAMSE_sal(imo, imd, sal,sal2)

% PAMSE - measure the distortion degree of distorted image 'imd' with the reference image 'imo'
% using the Gaussian smoothed MSE.
% 
% inputs:
% 
% imo - the reference image (grayscale image, double type, 0~255)
% imd - the distorted image (grayscale image, double type, 0~255)
% 
% outputs:

% score: the distortion scores obtained by PAMSE

% This is an implementation of the PAMSE algorithm in the following paper:
% W. Xue, X. Mou, L. Zhang, and X. Feng, "Perceptual Fidelity Aware Mean Squared Error," In ICCV 2013.

E_map = imo-imd;

%%% PAMSE (gaussian smoothed MSE) %%%%%%
sigma = 0.8;
h = fspecial('gaussian', (2*ceil(3*sigma)+1)*[1 1],sigma);
HErr = conv2(E_map,h,'valid');
%score = mean2(HErr.^2);
sal = imresize(sal, [size(HErr,1) size(HErr,2)]);
score_sal = sum(sum((HErr.^2).*sal)) / sum(sum(sal));
sal2 = imresize(sal2, [size(HErr,1) size(HErr,2)]);
score_sal2 = sum(sum((HErr.^2).*sal2)) / sum(sum(sal2));
