clc;clear;
close all;
%% PWRC computation
load('LIVE_SSIM');
LIVE_ssim = LIVE_ssim';
dmos = dmos';
pred = regress_hamid(LIVE_ssim,dmos);
% parameter preparation
p = opinion_norm(dmos,dmos_std);
p.flag = 0;     % DMOS -> flag 0; MOS -> flag 1;
p.act  = 1;     % enable/disable A(x,T): p.act->1/0 
th = 0:0.5:110; % customize observation interval

[PWRC_th,AUC] = PWRC(pred,dmos,th,p); 
disp(['The AUC value of SSIM is ',num2str(AUC)]);

delta_value = delta_MOS(pred,dmos,p);
disp(['The delta MOS value of SSIM is ',num2str(delta_value)]);
%% SA-ST curve
figure;
plot(th, PWRC_th, 'Color', 'r', 'LineWidth', 1.5, 'LineStyle', '-');
xlabel('\it{T}');
ylabel('PWRC');
legend('SSIM');
grid on;
title('\it{SA-ST} curve on LIVE II database');