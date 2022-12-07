% ------------------------------------------------------------------------------------------
% Confusing image quality assessment: Towards better augmented reality experience
% Huiyu Duan, Xiongkuo Min, Yucheng Zhu, Guangtao Zhai, Xiaokang Yang, and Patrick Le Callet
% IEEE Transactions on Image Processing (TIP)
% ------------------------------------------------------------------------------------------
% ------------------------------------------------------------------------------------------
% code to process captured mixed images, including "crop" and "save"
% ------------------------------------------------------------------------------------------

clc
clear all

%% process mixed images captured from Unity
path_mixed_img = '..\img_Mixed\raw_captured\';
path_mixed_write = '..\img_Mixed\cropped\';
path_mixed_write2 = '..\img_Mixed\captured\';

random_bg = load('random_bg.mat').random_bg;
random_index = load('random_index1.mat').random_index;

for cnt=1:560
    cnt
    img_name = dir([path_mixed_img,num2str(cnt),'__*.png']).name;
    img = imread([path_mixed_img,img_name]);
    r = centerCropWindow2d(size(img),[900,1440]);
    img_crop = imcrop(img,r);
    imwrite(img_crop, [path_mixed_write,num2str(random_index(cnt)),'.png']);
    imwrite(img, [path_mixed_write2,num2str(random_index(cnt)),'.png']);
end

%% process BG images captured from Unity
path_mixed_img = '..\img_BG\captured\';
path_mixed_write = '..\img_BG\cropped\';

random_bg = load('random_bg.mat').random_bg;
[~,reverse_index] = sort(random_bg);

for cnt=1:20
    cnt
    img_name = dir([path_mixed_img,num2str(cnt),'.png']).name;
    img = imread([path_mixed_img,img_name]);
    r = centerCropWindow2d(size(img),[900,1440]);
    img_crop = imcrop(img,r);
    imwrite(img_crop, [path_mixed_write,num2str(reverse_index(cnt)),'.png']);
end