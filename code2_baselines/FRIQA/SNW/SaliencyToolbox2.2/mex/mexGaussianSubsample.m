% mexGaussianSubsample - smooths and subsamples image (mex file).
%
% result = mexGaussianSubsample(image)
%    Convolves image with a 6x6 separable Gaussian kernel
%    and subsamples by a factor of two, all in one
%    integrated operation. 
%
% See also makeDyadicPyramid, makeSqrt2Pyramid, makeGaussianPyramid.

% This file is part of the SaliencyToolbox - Copyright (C) 2006-2008
% by Dirk B. Walther and the California Institute of Technology.
% See the enclosed LICENSE.TXT document for the license agreement. 
% More information about this project is available at: 
% http://www.saliencytoolbox.net

function result = mexGaussianSubsample(image)

mexFileLost;
