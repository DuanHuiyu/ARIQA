function ss = ltg_index(x1,x2)
%==========================================================================
% 1) Please cite the paper (K. Gu, G. Zhai, X. Yang, and W. Zhang, ¡°An 
% efficient color image quality metric with local-tuned-global model,¡± 
% in Proc. IEEE Int. Conf. Image Process., Oct. 2014.)
% 2) If any question, please contact me through guke.doctor@gmail.com; 
% gukesjtuee@gmail.com. 
% 3) Welcome to cooperation, and I am very willing to share my experience.
%==========================================================================
[m,n,l] = size(x1);
f = min(2,max(1,round(min(m,n)/256)));
if(f > 1)
lpf = ones(f,f);
lpf = lpf/sum(lpf(:));
x1 = imfilter(x1,lpf,'symmetric','same');
x2 = imfilter(x2,lpf,'symmetric','same');
x1 = x1(1:f:end,1:f:end,:);
x2 = x2(1:f:end,1:f:end,:);
end
Y1 = 0.299*double(x1(:,:,1))+0.587*double(x1(:,:,2))+0.114*double(x1(:,:,3));
Y2 = 0.299*double(x2(:,:,1))+0.587*double(x2(:,:,2))+0.114*double(x2(:,:,3));
I1 = 0.596*double(x1(:,:,1))-0.274*double(x1(:,:,2))-0.322*double(x1(:,:,3));
I2 = 0.596*double(x2(:,:,1))-0.274*double(x2(:,:,2))-0.322*double(x2(:,:,3));
Q1 = 0.211*double(x1(:,:,1))-0.523*double(x1(:,:,2))+0.312*double(x1(:,:,3));
Q2 = 0.211*double(x2(:,:,1))-0.523*double(x2(:,:,2))+0.312*double(x2(:,:,3));
dx = [3 0 -3;10 0 -10;3 0 -3]/16;
dy = dx';
xx1 = conv2(Y1, dx, 'same');
xy1 = conv2(Y1, dy, 'same');
gm1 = sqrt(xx1.^2 + xy1.^2);
xx2 = conv2(Y2, dx, 'same');
xy2 = conv2(Y2, dy, 'same');
gm2 = sqrt(xx2.^2 + xy2.^2);
Gm1 = 2*gm1.*gm2;
Gm2 = gm1.^2 + gm2.^2;
% Local
T1 = 300;
In = 0.03;
Th = 0.15;
GM = (Gm1+T1)./(Gm2+T1);
Gl = sort(GM(:));
y1 = mean2(Gl(1:round(Th*end)).^In);
% Global
T2 = 300;
In = 0.01;
Gg = (Gm1+T2)./(Gm2+T2);
y2 = mean2(Gg.^In);
% Color
T3 = 1000;
T4 = 1000;
In = 0.02;
IM = (2*I1.*I2+T3)./(I1.^2+I2.^2+T3);
QM = (2*Q1.*Q2+T4)./(Q1.^2+Q2.^2+T4);
y3 = real(mean2((IM.*QM).^In));
% Fuse
ss = y1/y2*y3;