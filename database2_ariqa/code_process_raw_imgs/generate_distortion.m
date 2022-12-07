% ------------------------------------------------------------------------------------------
% Confusing image quality assessment: Towards better augmented reality experience
% Huiyu Duan, Xiongkuo Min, Yucheng Zhu, Guangtao Zhai, Xiaokang Yang, and Patrick Le Callet
% IEEE Transactions on Image Processing (TIP)
% ------------------------------------------------------------------------------------------
% ------------------------------------------------------------------------------------------
% code to generate distortions for ariqa
% ------------------------------------------------------------------------------------------

img_size = [900 1440];

%%
path_raw_img = '..\img_AR\reference\';
path_write = '..\img_AR\distorted\';

for cnt = 1:20
    cnt
    % read current image
    img = im2double(imread([path_raw_img,num2str(cnt),'.png']))*255;
    imwrite(img/255,[path_write,num2str(cnt),'_0.png']);
    
    % jpeg
    jpeg_level = [7,3];
    for i = 1:size(jpeg_level,2)
        imwrite(img/255,[path_write,num2str(cnt),'_',num2str(i),'.jpg'],'Quality',jpeg_level(i));
    end
    
    % gaussian blur
    % gblur_level = [7,15,39,91];
    % for i=1:size(gblur_level,2)
    %     H = fspecial('gaussian', gblur_level(i), gblur_level(i)/6);
    %     img_blur = imfilter(img,H,'symmetric');
    %     imwrite(img_blur/255, [path_write,num2str(cnt),'_',num2str(i+4),'.png']);
    % end
    
    % resolution
    scale_level = [5,10];
    for i=1:size(scale_level,2)
        img_resize = imresize(img,1/scale_level(i));
        img_resize = imresize(img_resize,img_size);
        imwrite(img_resize/255, [path_write,num2str(cnt),'_',num2str(i+2),'.png']);
    end
    
    % img_contrast1 = cubic_F(coef1,img);
    % imwrite(img_contrast1/255, [path_write,num2str(cnt),'_10.png']);
    % img_contrast2 = cubic_F(coef2,img);
    % imwrite(img_contrast2/255, [path_write,num2str(cnt),'_20.png']);
    % img_contrast3 = logistic_F(coef3,img);
    % imwrite(img_contrast3/255, [path_write,num2str(cnt),'_30.png']);
    % img_contrast4 = logistic_F(coef4,img);
    % imwrite(img_contrast4/255, [path_write,num2str(cnt),'_40.png']);
    img_contrast5 = gamma_F(4,img);
    imwrite(img_contrast5/255, [path_write,num2str(cnt),'_5.png']);
    img_contrast8 = gamma_F(1/4,img);
    imwrite(img_contrast8/255, [path_write,num2str(cnt),'_6.png']);
end

%% different contrast adjustment methods
x1 = [0,127.5,255,9]';
y1 = [0,127.5,255,25]';
x2 = [0,127.5,255,15]';
y2 = [0,127.5,255,25]';
x3 = [0,127.5,255,25]';
y3 = [0,127.5,255,15]';
x4 = [0,127.5,255,25]';
y4 = [0,127.5,255,9]';
coef0 = [1,2,1,1];
[coef1 ehat J] = nlinfit(x1,y1,@cubic_F,coef0);
[coef2 ehat J] = nlinfit(x2,y2,@cubic_F,coef0);
[coef3 ehat J] = nlinfit(x3,y3,@logistic_F,coef0);
[coef4 ehat J] = nlinfit(x4,y4,@logistic_F,coef0);
hold on
xmax = max(x1);
xmin = min(y1);
xnum = 2*length(x1)+50;
figure (1)
xp = linspace(xmin,xmax,xnum);
yp1 = cubic_F(coef1,xp);
plot(xp,yp1,'g');
hold on
yp2 = cubic_F(coef2,xp);
plot(xp,yp2,'r');
hold on
yp3 = logistic_F(coef3,xp);
plot(xp,yp3,'b');
hold on
yp4 = logistic_F(coef4,xp);
plot(xp,yp4,'m');
hold on
plot(xp,xp,'k');

figure (2)
yp5 = gamma_F(2,xp);
plot(xp,yp5,'g');
hold on
yp6 = gamma_F(1.5,xp);
plot(xp,yp6,'r');
hold on
yp7 = gamma_F(1/1.5,xp);
plot(xp,yp7,'b');
hold on
yp8 = gamma_F(1/2,xp);
plot(xp,yp8,'m');
hold on
plot(xp,xp,'k');

figure (3)
yp9 = gamma_F(4,xp);
plot(xp,yp9,'g');
hold on
yp10 = gamma_F(1/4,xp);
plot(xp,yp10,'m');
hold on
plot(xp,xp,'k');


function F = cubic_F(alpha,x)
F = alpha(1)*x.^3+alpha(2)*x.^2+alpha(3)*x+alpha(4);
end

function F = logistic_F(beta,x)
F = (beta(1)-beta(2))./(1+exp(-((x-beta(3))/beta(4))))+beta(2);
end

function F = gamma_F(n,x)
F = ((255^((1/n)-1))*x).^n;
end