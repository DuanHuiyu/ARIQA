perSRCC(i) = corr(score(:,i),GT,'type','Spearman');
[delta,beta,yhat,y,diff] = findrmse2(score(:,i),GT);
score_mapped(:,i) = yhat;
perRMSE(i) = sqrt(sum(diff.^2)/length(diff));
perPLCC(i) = corr(GT, yhat, 'type','Pearson');