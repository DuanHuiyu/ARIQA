function result = mexConv2PreserveEnergy(data,filter)
% mexConv2PreserveEnergy - 2d convolution that avoids bleeding energy over the edge (mex file).
%
% result = mexConv2PreserveEnergy(data,filter)
%    Convolves data with the 2d (non-separable) filter.
%    At the boundary, the value of missing pixels is assumed
%    to be equal to the mean over the present pixels
%    to avoid border artefacts.
%
% See also sepConv2PreserveEnergy.

% This file is part of the SaliencyToolbox - Copyright (C) 2006-2008
% by Dirk B. Walther and the California Institute of Technology.
% See the enclosed LICENSE.TXT document for the license agreement. 
% More information about this project is available at: 
% http://www.saliencytoolbox.net

mexFileLost;
