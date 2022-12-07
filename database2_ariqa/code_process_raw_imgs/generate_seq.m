% ------------------------------------------------------------------------------------------
% Confusing image quality assessment: Towards better augmented reality experience
% Huiyu Duan, Xiongkuo Min, Yucheng Zhu, Guangtao Zhai, Xiaokang Yang, and Patrick Le Callet
% IEEE Transactions on Image Processing (TIP)
% ------------------------------------------------------------------------------------------
% ------------------------------------------------------------------------------------------
% code to control the sequence in Unity
% ------------------------------------------------------------------------------------------

% AR: 1-260  -1
% BG: 1-20  -1
% mixing: 1-4  -1

clc
clear all

% random_bg = 1:20;
% random_index = randperm(size(random_bg,2));
% random_bg = random_bg(:,random_index);
random_bg = load('random_bg.mat').random_bg;

%% experiment

cnt = 0;
seq = [];

for seq_mixing = 1:4
    for seq_ar = 1:140
        seq_bg = ceil(seq_ar/7);
        cnt = cnt+1;
        seq(:,cnt) = [seq_mixing-1,seq_ar-1,random_bg(seq_bg)-1];
    end
end

text_1 = [];
text_2 = [];
text_3 = [];
for cnt = 1:size(seq,2)
    text_1 = [text_1,num2str(seq(1,cnt)),','];    % mixing seq
    text_2 = [text_2,num2str(seq(2,cnt)),','];    % ar seq
    text_3 = [text_3,num2str(seq(3,cnt)),','];    % bg seq
end

% random seq
random_index = randperm(size(seq,2));
seq = seq(:,random_index);

% text to unity
text_random_1 = [];
text_random_2 = [];
text_random_3 = [];
for cnt = 1:size(seq,2)
    text_random_1 = [text_random_1,num2str(seq(1,cnt)),','];    % mixing seq
    text_random_2 = [text_random_2,num2str(seq(2,cnt)),','];    % ar seq
    text_random_3 = [text_random_3,num2str(seq(3,cnt)),','];    % bg seq
end