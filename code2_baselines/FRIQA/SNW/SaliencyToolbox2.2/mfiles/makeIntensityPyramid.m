% makeIntensityPyramid - creates an intensity pyramid.
%
% intPyr = makeIntensityPyramid(image,type)
%    Creates an intensity pyramid from image.
%       image: an Image structure for the input image.
%       type: 'dyadic' or 'sqrt2'
%
% See also makeFeaturePyramids, makeGaussianPyramid, dataStructures.

% This file is part of the SaliencyToolbox - Copyright (C) 2006-2008
% by Dirk B. Walther and the California Institute of Technology.
% See the enclosed LICENSE.TXT document for the license agreement. 
% More information about this project is available at: 
% http://www.saliencytoolbox.net

function intPyr = makeIntensityPyramid(image,type)

declareGlobal;

im = loadImage(image);

map.origImage = image;
map.label = 'Intensity';
map.data = mean(im,3);
map.date = timeString;

intPyr = makeGaussianPyramid(map,type);
