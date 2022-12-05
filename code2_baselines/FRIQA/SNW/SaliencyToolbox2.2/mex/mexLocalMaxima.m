% mexLocalMaxima - returns statistics over local maxima (mex file).
%
% [lm_avg,lm_num,lm_sum] = mexLocalMaxima(data,thresh)
%    Returns the average value (lm_avg), the number (lm_num),
%    and the sum (lm_sum) of local maxima in data that exceed thresh.
%
% See also maxNormalizeStd.

% This file is part of the SaliencyToolbox - Copyright (C) 2006-2008
% by Dirk B. Walther and the California Institute of Technology.
% See the enclosed LICENSE.TXT document for the license agreement. 
% More information about this project is available at: 
% http://www.saliencytoolbox.net

function [lm_avg,lm_num,lm_sum] = mexLocalMaxima(data,thresh)

mexFileLost;
