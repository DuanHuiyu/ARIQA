PWRC source code release.
Author  : Qingbo Wu
Version : Beta1.0

The authors are with the School of Information and Communication Engineeri-
ng, University of Electronic Science and Technology of China.

This research aims to develop more reasonable and reliable rank correlation
indicator (RCI) to evaluate different IQA models. When two different ranki-
ng results share the same spearman's or kendall's rank correlation coeffic-
ients, we intuitively illustrate that they could recommend completely diff-
erent enhancement results. Both the objective and subjective tests confirm 
that a user-friendly RCI tends to reward the capability of correctly ranki-
ng high-quality images and suppress the attention towards insensitive rank 
mistakes.
   

===========================================================================

-------------------------COPYRIGHT NOTICE----------------------------------
Copyright (c) 2018 University of Electronic Science and Technology of China
All rights reserved.

For researchers and educators who wish to use the code for non-commercial 
research and/or educational purposes, we can provide access under the fo-
llowing terms:

1. Researcher shall use the code only for non-commercial research and ed-
   ucational purposes.
2. Researcher accepts full responsibility for his or her use of the code
   and shall defend and indemnify University of Electronic Science and T-
   echnology of China, including their employees, Trustees, officers and 
   agents, against any and all claims arising from Researcher's use of t-
   he code.
3. Researcher may provide research associates and colleagues with access 
   to the code provided that they first agree to be bound by these terms 
   and conditions.
4. University of Electronic Science and Technology of China reserves the 
   right to terminate Researcher's access to the code at any time.
5. If Researcher is employed by a for-profit, commercial entity, Researc-
   her's employer shall also be bound by these terms and conditions, and 
   Researcher hereby represents that he or she is fully authorized to en-
   ter into this agreement on behalf of such employer.

---------------------------Instructions------------------------------------
This is a MATLAB implementation of the perceptually weighted rank correlat-
ion (PWRC) indicator. If this code is helpful for your research, please ci-
te the following paper in your bibliography, i.e.,
1. Q. Wu, H. Li, F. Meng and K. N. Ngan, "A Perceptually Weighted Rank Cor-
   relation Indicator for Objective Image Quality Assessment", IEEE Transa-
   ctions on Image Processing, In Press, 2018.

Files: 

LIVE_SSIM.mat: SSIM scores computed on LIVE II database
https://ece.uwaterloo.ca/~z70wang/research/iwssim/

./DB_image:
fig_lena_org:     Original image for Lena
fig_lena_block:   Image compressed by H.264/AVC encoder (QP=47)
fig_lena_rank1~5: Deblocked versions of fig_lena_block.
Deblocking software: http://www.cs.tut.fi/~foi/SA-DCT/#ref_software  

Usage: 

1. The script demo.m shows how to compute the PWRC between the predicted i-
   mage quality scores and the DMOS. The vector representation PWRC_th is
   used for drawing the SA-ST curve. The scalar representation AUC is used 
   for quantitative comparison;

2. The script rational_test.m reproduces the image fusion results for the 
   predicted ranks S8 and S9. In this example, S8 and S9 are considered eq-
   uivalent to each other in terms of SRCC and KRCC. But, S8 produces bett-
   er image fusion result than S9, which is correctly measured by our PWRC.
   More explanations could refer to Figs. 6-7 in our paper.  

Notice:
We have tested our software under the Windows 7-64bit OS. This is a vani-
lla version. If you have any suggestions or corrections in the usage of 
this code, please feel free to contact wqb.uestc@gmail.com
