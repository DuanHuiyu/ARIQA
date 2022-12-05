
clear;clc;

ref = imread('ref.bmp');
ref = double(ref(:,:,1));

dist = imread('dist.bmp');
dist = double(dist(:,:,1));
vifp = vifp_mscale(ref,dist)

dist2 = imread('dist2.bmp');
dist2 = double(dist2(:,:,1));
vifp2 = vifp_mscale(ref,dist2)