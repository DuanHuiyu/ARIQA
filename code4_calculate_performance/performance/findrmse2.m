
function [delta,beta,yhat,y,diff] = findrmse2(iqa,mos)
x = iqa;
y = mos;

z = find(isinf(x)==1);
x(z) = [];
y(z) = [];

temp = corrcoef(x,y);
if (temp(1,2)>0)
beta0(3) = mean(x);
beta0(1) = abs(max(y) - min(y));
beta0(4) = mean(y);
beta0(2) = 1/std(x);
beta0(5) = 1;
else
beta0(3) = mean(x);
beta0(1) = -abs(max(y) - min(y));
beta0(4) = mean(y);
beta0(2) = 1/std(x);
beta0(5) = 1;
end

maxiter = 10000;
[beta ehat J] = nlinfit(x,y,@myfunn3,beta0,maxiter);
[yhat delta] = nlpredci(@myfunn3,x,beta,ehat,J);
diff = abs(y - yhat);
