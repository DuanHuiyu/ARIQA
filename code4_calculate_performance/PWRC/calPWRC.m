function AUC = calPWRC(pred,mos,mos_std)

% pred = pred'; % in case the "pred" is a row vector
% mos = mos'; % in case the "mos" is a row vector
pred = regress_hamid(pred,mos);
% parameter preparation
p = opinion_norm(mos,mos_std);
p.flag = 0;     % DMOS -> flag 0; MOS -> flag 1;
p.act  = 1;     % enable/disable A(x,T): p.act->1/0 
th = 0:0.5:110; % customize observation interval

[PWRC_th,AUC] = PWRC(pred,mos,th,p); 

end