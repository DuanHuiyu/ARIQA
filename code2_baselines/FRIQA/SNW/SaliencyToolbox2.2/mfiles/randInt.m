% randInt - create random integers.
%
% r = randInt(N);
%   returns a random uniformly distributed integer 1 <= r <= N.
%
% r = randInt([M N]);
%   returns a random uniformly distributed integer M <= r <= N.
%
% r = randInt(M,sz) and r = randInt([M N],sz)
%   returns an array of size sz with random integers.
%
% If sz is a scalar, a square array of size [sz sz] is returned.

% This file is part of the SaliencyToolbox - Copyright (C) 2006-2008
% by Dirk B. Walther and the California Institute of Technology.
% See the enclosed LICENSE.TXT document for the license agreement. 
% More information about this project is available at: 
% http://www.saliencytoolbox.net

function r = randInt(arg,varargin)

if (isempty(varargin))
  sz = 1;
else
  sz = varargin{1};
end


if (length(arg) < 2)
  m = 1;
  n = arg(1);
else
  m = arg(1);
  n = arg(2);
end

r = floor(rand(sz)*(n-m+1) + m);
