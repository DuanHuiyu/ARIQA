% ========================================================================
% IGM Index with automatic downsampling, Version 1.0
% Copyright(c) 2012 Jinjian Wu, Weisi Lin, Guangming Shi, and Anmin Liu
% All Rights Reserved.
%
% ----------------------------------------------------------------------
% Permission to use, copy, or modify this software and its documentation
% for educational and research purposes only and without fee is here
% granted, provided that this copyright notice and the original authors'
% names appear on all copies and supporting documentation. This program
% shall not be used, rewritten, or adapted as the basis of a commercial
% software or hardware product without first obtaining permission of the
% authors. The authors make no representations about the suitability of
% this software for any purpose. It is provided "as is" without express
% or implied warranty.
%----------------------------------------------------------------------
%
% This is an implementation of the algorithm for calculating the
% IGM based IQA index between two images.
%
% Please refer to the following paper
%
% J. Wu, W. Lin, G. Shi, A. Liu, “A Perceptual Quality Metric with 
% Internal Generative Mechanism”, IEEE Trans. on Image Processing, 
% accepted, 2012
%-----------------------------------------------------------------------
%
% Usage:
% Given 2 test images img1 and img2. For gray-scale images, their dynamic 
% range should be 0-255.
% igm_val = func_igm_iqa_metric( img_ref, img_dst )
%-----------------------------------------------------------------------

clear all;
close all;
clc;


img_ref = imread( 'I01.bmp' );
img_dst = imread( 'I01_01_2.bmp' );

igm_val = func_igm_iqa_metric( img_ref, img_dst );
fprintf( 'igm_value = %.4f\n', igm_val );

% end of this file