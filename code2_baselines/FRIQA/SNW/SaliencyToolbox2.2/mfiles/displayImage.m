% displayImage - displays an image in a convenient way in the current axes.
%
% displayImage(img) - displays image in a new window
%    img can be of any numerical type or a logical, and it
%        must have two (gray-scale) or three (RGB) dimensions.
%    img can be an Image structure (see initializeImage).
%    The image is scaled appropriately.
%
% displayImage(img,doNormalize)
%    If doNormalize is 1, the image is maximum-normalized 
%    (default: 0).
%
% See also displayMap, displayMaps, showImage, initializeImage, dataStructures.

% This file is part of the SaliencyToolbox - Copyright (C) 2006-2008
% by Dirk B. Walther and the California Institute of Technology.
% See the enclosed LICENSE.TXT document for the license agreement. 
% More information about this project is available at: 
% http://www.saliencytoolbox.net

function displayImage(img,doNormalize)

if (nargin < 2)
  doNormalize = 0;
end

if (isa(img,'struct'))
  displayImage(loadImage(img),doNormalize);
  return;
end

if (isa(img,'logical'))
  img = double(img);
end

if (~isa(img,'double'))
  img = im2double(img);
end

img = squeeze(img);
num_dims = length(size(img));

if ((num_dims ~= 2) & (num_dims ~= 3))
  disp([mfilename ' error - unknown image format: ' class(img)]);
  return;
end

mx = max(img(:));
mn = min(img(:));

if (doNormalize & (mx > mn))
  img = mat2gray(img);
  %img = (img - mn) / (mx - mn);
end

if (num_dims == 2)
  % gray scale image -> RGB
  img = reshape(img,size(img,1),size(img,2),1);
  img(:,:,2) = img(:,:,1);
  img(:,:,3) = img(:,:,1);
end

mx = max(max(max(img))); mn = min(min(min(img)));
if ((mx > 1.0) | (mn < 0.0))
  disp('showImage.m: image out of range 0.0 ... 1.0');
  fprintf('(max: %g; min: %g)\n',mx,mn);
  disp ('cutting off ...');
  img(find(img > 1.0)) = 1.0;
  img(find(img < 0.0)) = 0.0;
end

% display the RGB image
image(img);
axis image;
