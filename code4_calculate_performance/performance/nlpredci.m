function [ypred, delta] = nlpredci(model,X,beta,resid,J,alpha,simopt,predopt)
%NLPREDCI Confidence intervals for nonlinear least squares prediction.
%   [YPRED, DELTA] = NLPREDCI(FUN,X,BETA,RESID,J) returns predictions
%   (YPRED) and 95% confidence interval half-widths (DELTA) for the
%   function F at input values X.  Before using this function you use
%   NLINFIT to fit FUN by non-linear least squares and get estimated
%   coefficient values BETA, residuals RESID, and Jacobian J.
%
%   [YPRED, DELTA] = NLPREDCI(FUN,X,BETA,RESID,J,ALPHA,SIMOPT,PREDOPT)
%   provides control over the confidence bounds.  ALPHA defines the
%   confidence level as 100(1-ALPHA) percent, and has a default of 0.05.
%   SIMOPT can be 'on' for simultaneous confidence bounds or 'off' (the
%   default) for non-simultaneous bounds.  PREDOPT can be 'curve' (the
%   default) for confidence intervals for the estimated curve (function
%   value) at X or 'observation' for confidence intervals for a new
%   observation at X.
%
%   The confidence interval calculation is valid for systems where the 
%   length of RESID exceeds the length of BETA and J has full column rank
%   at BETA.
%
%   Example:
%      [beta,resid,J]  = nlinfit(input,output,@f,betainit);
%      [yp, ci] = nlpredci(@f,newx,beta,resid,J);
%
%   See also NLINFIT, NLPARCI, NLINTOOL.

%   Bradley Jones 1-28-94
%   Copyright 1993-2000 The MathWorks, Inc. 
% $Revision: 2.16 $  $Date: 2000/05/26 18:53:21 $

%initialization
if nargin < 5
   error('Requires five inputs. Usage: [P, D] = NLPREDCI(''F'',X,BETA,RESID,J)');
end;
if (nargin<6 | isempty(alpha)), alpha = 0.05; end         % 95% conf intervals
if (nargin<7 | isempty(simopt)), simopt = 'off'; end      % not simultaneous
if (nargin<8 | isempty(predopt)), predopt = 'curve'; end  % estimate curve
if (length(alpha)~=1 | alpha<=0 | alpha >= 1)
   error('ALPHA must be a scalar satisfying 0<ALPHA<1.');
end
switch(simopt)
 case 'on', dosim = 1;
 case 'off', dosim = 0;
 otherwise, error('SIMOPT must be ''on'' or ''off''.');
end
switch(predopt)
 case {'c' 'curve'}, newobs = 0;
 case {'o' 'observation'}, newobs = 1;
 otherwise, error('PREDOPT must be ''curve'' or ''observation''.');
end

resid = resid(:);
[m,n] = size(J);
if m <= n
   error('The number of observations must exceed the number of parameters.');
end;

if length(beta) ~= n
   error('The length of BETA must equal the number of columns in J.')
end;

% odds are, an input of length m should be a column vector
if (size(X,1)==1 & size(X,2)==m), X = X(:); end

% approximation when a column is zero vector
temp = find(max(abs(J)) == 0);
if ~isempty(temp)
   J(temp,:) = J(temp,:) + sqrt(eps);
end;

%calculate covariance
[Q, R] = qr(J,0);
Rinv = R\eye(size(R));

ypred = feval(model, beta, X);

delta = zeros(size(X,1),length(beta));

for i = 1:length(beta)
   change = zeros(size(beta));
   if (beta(i) == 0)
      nb = sqrt(norm(beta));
      change(i) = sqrt(eps) * (nb + (nb==0));
   else
      change(i) = sqrt(eps)*beta(i);
   end
   predplus = feval(model, beta+change, X);
   delta(:,i) = (predplus - ypred)/change(i);
end

E = delta*Rinv;
if (newobs)
   delta = sqrt(1 + sum(E.*E,2));
else
   delta = sqrt(sum(E.*E,2));
end   

v = m-n;
rmse = sqrt(sum(resid.*resid)/v);

% Calculate confidence interval
if (dosim)
   crit = sqrt(n * finv(1-alpha, n, v));
else
   crit = tinv(1-alpha/2,v);
end

delta = delta .* rmse * crit;
