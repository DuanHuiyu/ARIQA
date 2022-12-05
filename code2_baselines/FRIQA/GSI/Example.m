% GSM Compute the quality of a image with respective to its reference
%   Quality=GSM(ref_img,dst_img,Pmasking,Pfusion)
%     Quality : the quality score, 0~1
%     ref_img : reference image, (0~255,double type)
%     dst_img : distorted image, (0~255,double type)
%     Pmasking: masking parameter, optional, 0~inf,default value is 200
%     Pfusion : fusion parameter, optional, 0~1,default value is 0.1

close all;clc;
I1=double(imread('src.bmp'));
I2=double(imread('dst.bmp'));
GSM(I1,I2)