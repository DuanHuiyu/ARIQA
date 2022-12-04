% ------------------------------------------------------------------------------------------
% Confusing image quality assessment: Towards better augmented reality experience
% Huiyu Duan, Xiongkuo Min, Yucheng Zhu, Guangtao Zhai, Xiaokang Yang, and Patrick Le Callet
% IEEE Transactions on Image Processing (TIP)
% ------------------------------------------------------------------------------------------

clc
close all
clear all

path = '..\database2_ariqa\subjective_data\';
subject_name = {'duan_20210725202708_1_560_False_False','duan_20210726140247_483_560_False_False';
    'min_20210726154441_1_280_True_True','min_20210726161009_281_560_False_False';
    'gao_20210726185828_1_280_True_True','gao_20210726193553_281_560_False_False';
    'wusijing_20210727192708_1_280_False_False','wusijing_20210727195236_281_560_False_False';
    'tenglong_20210728104024_1_280_False_False','tenglong_20210728105739_281_560_False_False';
    'tianyuan_20210728135322_1_280_True_True','tianyuan_20210728141710_281_560_False_False';
    'yangcheng_20210728145125_1_280_True_True','yangcheng_20210728152628_281_560_False_False';
    'zhangzicheng_20210728161314_1_280_True_True','zhangzicheng_20210728163034_281_560_False_False';
    'liyunhao_20210728180455_1_280_True_True','liyunhao_20210728183221_281_560_False_False';};
subject_num = size(subject_name,1);

subject_name2 = {'tudanyang_20210728213504_1_280_True_True','tudanyang_20210728215826_281_560_False_False';
    'zhuyucheng_20210729093018_1_280_True_True','zhuyucheng_20210729100443_281_560_False_False';
    'yangmingyang_20210729104423_1_280_True_True','yangmingyang_20210729110414_281_560_False_False';
    'shenwang_20210729122454_1_280_False_False','shenwang_20210729124022_281_560_False_False';
    'laoduan_20210729160531_1_280_True_True','laoduan_20210729200939_281_560_True_True';
    'jiajun_20210729184006_1_280_True_True','jiajun_20210729190739_281_560_False_False';
    'sunwei_20210817145615_1_280_True_True','sunwei_20210817153159_281_560_False_False';};
subject_num2 = size(subject_name2,1);

subject_name3 = {'geqihang_20210813104958_1_280_True_True','geqihang_20210813123211_281_560_False_False';
    'wugeer_20210801101822_1_280_False_False','wugeer_20210801104910_281_560_False_False';
    'niuniu_20210815205007_1_280_False_False','niuniu_20210816142228_281_560_False_False';
    'pengyicong_20210813161431_1_280_True_True','pengyicong_20210813163339_281_560_False_False';
    'yifuwang_20210816201124_1_280_False_False','yifuwang_20210816202913_281_560_False_False';
    'shifangyu_20210816205753_1_280_True_True','shifangyu_20210816212336_281_560_False_False';};
subject_num3 = size(subject_name3,1);

for cnt = 1:subject_num
    cnt
    for part = 1:2
        [numeric,text,raw] = xlsread([path,char(subject_name(cnt,part)),'\IQA.csv']);
        for row = 1:size(numeric,1)
            scores_raw(numeric(row,1),cnt) = numeric(row,2);
            text_raw(numeric(row,1),cnt) = text(row,2);
        end
    end
end
random_index = load('..\database2_ariqa\code_process_raw_imgs\random_index1.mat').random_index;
[~,reverse_index] = sort(random_index);
scores_sorted1 = scores_raw(reverse_index,:);
text_sorted1 = text_raw(reverse_index,:);

for cnt = 1:subject_num2
    cnt
    for part = 1:2
        [numeric,text,raw] = xlsread([path,char(subject_name2(cnt,part)),'\IQA.csv']);
        for row = 1:size(numeric,1)
            scores_raw2(numeric(row,1),cnt) = numeric(row,2);
            text_raw2(numeric(row,1),cnt) = text(row,2);
        end
    end
end
random_index = load('..\database2_ariqa\code_process_raw_imgs\random_index2.mat').random_index;
[~,reverse_index] = sort(random_index);
scores_sorted2 = scores_raw2(reverse_index,:);
text_sorted2 = text_raw2(reverse_index,:);

for cnt = 1:subject_num3
    cnt
    for part = 1:2
        [numeric,text,raw] = xlsread([path,char(subject_name3(cnt,part)),'\IQA.csv']);
        for row = 1:size(numeric,1)
            if numeric(row,1)>0
                scores_raw3(numeric(row,1),cnt) = numeric(row,2);
                text_raw3(numeric(row,1),cnt) = text(row,2);
            end
        end
    end
end
random_index = load('..\database2_ariqa\code_process_raw_imgs\random_index3.mat').random_index;
[~,reverse_index] = sort(random_index);
scores_sorted3 = scores_raw3(reverse_index,:);
text_sorted3 = text_raw3(reverse_index,:);

scores_sorted = [scores_sorted1,scores_sorted2,scores_sorted3];

%% MOS
% ------------------compulate MOS with outliers removed--------------------
scoresOri = scores_sorted;
SCORE = scoresOri;%ceil(scoresOri.*2);

scoreMeanImg = zeros(size(SCORE));
scoreStdImg = zeros(size(SCORE));
scoreMeanPerson = zeros(size(SCORE));
scoreStdPerson = zeros(size(SCORE));

% mean score for each image (another words: /(subjects' number)) (MOS)
for img = 1:size(SCORE,1)
    scoreMeanImg(img,:) = mean(SCORE(img,:));
    scoreStdImg(img,:) = std(SCORE(img,:));
end

% mean score for each subject (another words: /(images' number))
for person = 1:size(SCORE,2)
    scoreMeanPerson(:,person) = mean(SCORE(:,person));
    scoreStdPerson(:,person) = std(SCORE(:,person));
end

% remove outliers
reject = zeros(1,size(SCORE,2));
beta = kurtosis(SCORE')';
for person = 1:size(SCORE,2)
    P = 0;
    Q = 0;
    for img = 1:size(SCORE,1)
        if (beta(img)>=2)&&(beta(img)<=4)
            k = 2;
        else
            k = sqrt(20);
        end
        if SCORE(img,person) >= scoreMeanImg(img,person) + k*scoreStdImg(img,person)
            P = P+1;
        elseif SCORE(img,person) <= scoreMeanImg(img,person) - k*scoreStdImg(img,person)
            Q = Q+1;
        end
    end
    if ((P+Q)/size(SCORE,1)>0.05) && (abs((P-Q)/(P+Q))<0.3)
        reject(1,person) = 1;
    end
end
accept = ~reject;

for j = 1:size(SCORE,2)
    if accept(j)
        kk = ((beta>=2)&(beta<=4))*2 + ((beta<2)|(beta>4))*sqrt(20);
        acceptIndex(:,j) = (SCORE(:,j)>=(mean(SCORE,2)-kk.*std(SCORE,[],2))).*(SCORE(:,j)<=(mean(SCORE,2)+kk.*std(SCORE,[],2)));
    else
        acceptIndex(:,j) = 0;
    end
end
acceptIndex = logical(acceptIndex);
acceptSubjectIndex = acceptIndex(:,find(accept==1));
acceptRatio = 1-sum(sum(acceptSubjectIndex))/(size(acceptSubjectIndex,1)*size(acceptSubjectIndex,2));
for i = 1:size(SCORE,1)
    MOS(i,1) = mean(SCORE(i,acceptIndex(i,:)));
    SD(i,1) = std(SCORE(i,acceptIndex(i,:)));
    num_obs(i,1) = sum(acceptIndex(i,:));
end


SCOREz = (SCORE - scoreMeanPerson)./scoreStdPerson;
SCOREz = (SCOREz+3)*100/6;
for i = 1:size(SCOREz,1)
    MOSz(i,1) = mean(SCOREz(i,acceptIndex(i,:)));
end

%% Distribution of subjectibe quality scores in the database
numberTotal = zeros(1,10);
scoresAfterRemoveOutliers = SCORE.*acceptIndex;
for i = 1:subject_num
    for j = 1:10
        numberTotal(j) = numberTotal(j)+length(find(scoresAfterRemoveOutliers(:,i)==j));
    end
end
figure(1)
bar(numberTotal);
axis([0 11 0 1600]);
xlabel('scores');
ylabel('Image Number');
title('Distribution of subjectibe quality scores');

%% Distribution of MOSs
figure(2)
hist(MOSz,13);
h = findobj(gca,'Type','patch');
axis([20 95 0 100]);
set(h,'facecolor',[0 0.4470 0.7410]);
set(h,'edgeColor','k');
xlabel('MOS');
ylabel('Image Number');
grid on
% title('Distribution of MOSs');

%% Process and analyze
for cnt = 1:560
    i = ceil((mod((cnt-1),140)+1)/7);
    j = mod((mod((cnt-1),140)),7);
    k = floor((cnt-1)/140); 
    level1 = k*7;   % alpha
    level2 = j+1;   % distortion
    level(cnt,:) = level1+level2;
    level_dis(cnt,:) = level2;
    scenes(cnt,:) = i;
    scores = MOSz;
end
figure(3)
hold on
scatter(level(find(scenes<=8),:),scores(find(scenes<=8),:),35,'ro')
scatter(level(find((scenes>8)&(scenes<=16)),:),scores(find((scenes>8)&(scenes<=16)),:),35,'go')
scatter(level(find(scenes>16),:),scores(find(scenes>16),:),35,'bo')
ylim([0 100])
xlim([0 30])
set(gca,'fontname','Times New Roman','Fontsize',16)
xlabel('level','fontname','Times New Roman','fontsize',20);  
ylabel('MOS','fontname','Times New Roman','fontsize',20);  
box on

figure(4)
hold on
scatter(level(find((level_dis<6)&(scenes<=8)),:),scores(find((level_dis<6)&(scenes<=8)),:),35,'ro')
scatter(level(find((level_dis<6)&(scenes>8)&(scenes<=16)),:),scores(find((level_dis<6)&(scenes>8)&(scenes<=16)),:),35,'go')
scatter(level(find((level_dis<6)&(scenes>16)),:),scores(find((level_dis<6)&(scenes>16)),:),35,'bo')
ylim([0 100])
xlim([0 27])
set(gca,'fontname','Times New Roman')%,'Fontsize',16)
legend('Webpage','Natural','Graphic');
xticks([1 2 3 4 5 8 9 10 11 12 15 16 17 18 19 22 23 24 25 26])
xticklabels({'Raw','JPEG1','JPEG2','Scaling1','Scaling2','Raw','JPEG1','JPEG2','Scaling1','Scaling2','Raw','JPEG1','JPEG2','Scaling1','Scaling2','Raw','JPEG1','JPEG2','Scaling1','Scaling2'})
xtickangle(60)
%xlabel('level','fontname','Times New Roman','fontsize',20);  
ylabel('MOS','fontname','Times New Roman')%,'fontsize',20);  
grid on
box on


for cnt = 1:560
    i = ceil((mod((cnt-1),140)+1)/7);
    j = mod((mod((cnt-1),140)),7);
    k = floor((cnt-1)/140); 
    level1 = k*7;   % alpha
    level2 = j+1;   % distortion
    if (level1+level2 == 1)||(level1+level2 == 8)||(level1+level2 == 15)||(level1+level2 == 22)
        level(cnt,:) = level1+level2+2;
    elseif (level1+level2 == 6)||(level1+level2 == 13)||(level1+level2 == 20)||(level1+level2 == 27)
        level(cnt,:) = level1+level2-5;
    elseif (level1+level2 == 7)||(level1+level2 == 14)||(level1+level2 == 21)||(level1+level2 == 28)
        level(cnt,:) = level1+level2-2;
    else
        level(cnt,:) = level1+level2;
    end
    level_k(cnt,:) = k;
    level_dis(cnt,:) = level2;
    scenes(cnt,:) = i;
    scores = MOSz;
end
figure(5)
hold on
scenes_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20];
% scenes_list = [2,5,8,10,13,16,18,19];
for i = 1:20
    for j = 0:3
        alg = find(((level_dis==1)|(level_dis==6)|(level_dis==7))&(level_k==j)&(scenes==scenes_list(i)));
        temp_level = level(alg,:);
        temp_score = scores(alg,:);
        if scenes_list(i)<=8
            plot([temp_level(2,1),temp_level(1,1),temp_level(3,1)],[temp_score(2,1),temp_score(1,1),temp_score(3,1)],'-*r');
        elseif (scenes_list(i)>8)&&(scenes_list(i)<=16)
            plot([temp_level(2,1),temp_level(1,1),temp_level(3,1)],[temp_score(2,1),temp_score(1,1),temp_score(3,1)],'-*g');
        else
            plot([temp_level(2,1),temp_level(1,1),temp_level(3,1)],[temp_score(2,1),temp_score(1,1),temp_score(3,1)],'-*b');
        end
        hold on
    end
end
ylim([0 100])
xlim([0 27])
%%
figure(6)
hold on
scenes_list = [1,4,8,10,13,16,18,19];
for i = 1:8
    for j = 0:3
        alg = find(((level_dis==1)|(level_dis==6)|(level_dis==7))&(level_k==j)&(scenes==scenes_list(i)));
        temp_level = level(alg,:);
        temp_score = scores(alg,:);
        if scenes_list(i)<=8
            h1 = plot([temp_level(2,1),temp_level(1,1),temp_level(3,1)],[temp_score(2,1),temp_score(1,1),temp_score(3,1)],'-*r');
        elseif (scenes_list(i)>8)&&(scenes_list(i)<=16)
            h2 = plot([temp_level(2,1),temp_level(1,1),temp_level(3,1)],[temp_score(2,1),temp_score(1,1),temp_score(3,1)],'-*g');
        else
            h3 = plot([temp_level(2,1),temp_level(1,1),temp_level(3,1)],[temp_score(2,1),temp_score(1,1),temp_score(3,1)],'-*b');
        end
        hold on
    end
end
ylim([0 100])
xlim([0 27])
set(gca,'fontname','Times New Roman')%,'Fontsize',16)
legend([h1,h2,h3],'Webpage','Natural','Graphic');
% legend('Webpage','Natural','Graphic');
xticks([1 3 5 8 10 12 15 17 19 22 24 26])
xticklabels({'Contrast(N)','Raw','Contrast(P)','Contrast(N)','Raw','Contrast(P)','Contrast(N)','Raw','Contrast(P)','Contrast(N)','Raw','Contrast(P)'})
xtickangle(45)
%xlabel('level','fontname','Times New Roman','fontsize',20);  
ylabel('MOS','fontname','Times New Roman')%,'fontsize',20);  
grid on
box on
