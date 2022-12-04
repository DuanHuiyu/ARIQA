function [ output ] = GP_overlap( p1,p2 )
%OVERLAP Summary of this function goes here
%   p1: [sigma1, mu1] 2-D parameters of Gaussian process;
%   p2: [sigma2, mu2] 2-D parameters of Gaussian process;
%   for example
%   p1=[8,27];
%   p2=[12,42];

const1 = 1/(p1(1)*sqrt(2*pi));
const2 = 1/(p2(1)*sqrt(2*pi));
conf_int1 = [p1(2)-2*p1(1),p1(2)+2*p1(1)];
conf_int2 = [p2(2)-2*p2(1),p2(2)+2*p2(1)];
x = min([conf_int1,conf_int2]):0.1:max([conf_int1,conf_int2]);
y1 = const1*exp(-(x-p1(2)).^2/(2*p1(1)^2));
y2 = const2*exp(-(x-p2(2)).^2/(2*p2(1)^2));
output = sum(min([y1;y2]))/sum(max([y1;y2]));

% for plot
%{
conf1 = (p1(2)-2*p1(1)):0.1:(p1(2)+2*p1(1));
conf2 = (p2(2)-2*p2(1)):0.1:(p2(2)+2*p2(1));
y_over = min([y1;y2]);
reg1 = const1*exp(-(conf1-p1(2)).^2/(2*p1(1)^2));
reg2 = const2*exp(-(conf2-p2(2)).^2/(2*p2(1)^2));

x1 = p1(2)-25:0.1:p1(2)+25;
x2 = p2(2)-30:0.1:p2(2)+30;
y1 = const1*exp(-(x1-p1(2)).^2/(2*p1(1)^2));
y2 = const2*exp(-(x2-p2(2)).^2/(2*p2(1)^2));

figure(1);
hold on;
fill([conf_int1(1),conf1,conf_int1(2)],[0,reg1,0],[1 0.9 0.9], 'EdgeColor','none');
fill([conf_int2(1),conf2,conf_int2(2)],[0,reg2,0],[1 0.9 0.9], 'EdgeColor','none');
fill([(p1(2)-2*p1(1)),x,(p2(2)+2*p2(1))],[0,y_over,0],[0.6 0.6 1], 'EdgeColor','none');
h1 = plot(x1,y1,'-r');
h2 = plot(x2,y2,'-b');
hold off;
grid on;
legend([h1,h2],'GP_{*}','GP_{*^{,}}');
%}

end


