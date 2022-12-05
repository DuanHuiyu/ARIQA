% initializeGlobal - initializes global variables
%
% initializeGlobal initializes the following global variables:
%    IS_INITIALIZED is set to 1 as a flag that initializeGlobal was called.
%    PD is set to the appropriate path delimiter for your operating system
%       ('/' for Unix variants, '\' for MS Windows).
%    IMG_EXTENSIONS is a cell arrays with possible extensions for image files.
%    DEBUG_FID is the file identifier for debugMsg output. It is by default
%       set to 0. Set to 1 for stdout or to the fid of a text file that
%       is open for write access.
%    BASE_DIR is set to ''.
%    IMG_DIR is set to ''.
%    DATA_DIR is set to ''.
%    TMP_DIR is set to the Matlab standard tempdir.
%
% In the SaliencyToolbox, all paths to images are assumed to be relative
% to IMG_DIR, and all paths to data files relative to DATA_DIR. If no
% base directory is supplied at the call of initializeGlobal, all these
% paths are empty, and all image paths will have to be absolute paths
% or relative to the current directory. Setting a base path allows for
% more flexibility in migrating data to other locations in the path,
% because paths will be relative to the base path, whch can be different
% on different machines, for example.
%
% initializeGlobal(base_dir) does the same as above, except:
%    BASE_DIR is set to base_dir and created if it doesn't exist.
%    IMG_DIR is set to base_dir/img and created if it doesn't exist.
%    DATA_DIR is set to base_dir/data and created if it doesn't exist.
%    TMP_DIR is set to base_dir/tmp and created if it doesn't exist.
%
% See also declareGlobal, ensureDirExists, debugMsg.

% This file is part of the SaliencyToolbox - Copyright (C) 2006-2008
% by Dirk B. Walther and the California Institute of Technology.
% See the enclosed LICENSE.TXT document for the license agreement. 
% More information about this project is available at: 
% http://www.saliencytoolbox.net

function initializeGlobal(varargin)

global IS_INITIALIZED;
IS_INITIALIZED = 1;

declareGlobal;

dbstop if error;

% this is the path delimiter set according to your OS
switch (computer)
  case 'PCWIN'
    PD = '\';
  case 'MAC'
    PD = '/';  % OS X uses '/' instead of ':'
  otherwise
    PD = '/';
end

if nargin >= 1
  BASE_DIR = [varargin{1} PD];     ensureDirExists(BASE_DIR);
  IMG_DIR = [BASE_DIR 'img' PD];   ensureDirExists(IMG_DIR);
  DATA_DIR = [BASE_DIR 'data' PD]; ensureDirExists(DATA_DIR);
  TMP_DIR = [BASE_DIR 'tmp' PD];   ensureDirExists(TMP_DIR);
else
  BASE_DIR = '';
  IMG_DIR = '';
  DATA_DIR = '';
  TMP_DIR = tempdir; ensureDirExists(TMP_DIR);
end


IMG_EXTENSIONS = {'*.pgm','*.ppm','*.tif','*.tiff','*.TIF',...
                  '*.jpg','*.JPG','*.jpeg','*.png','*.PNG',...
                  '*.gif','*.GIF','*.JPEG','*.PGM','*.PPM',...
                  '*.bmp','*.BMP'};

DEBUG_FID = 0;
DBW_FID = 240.75;

fprintf('\nSaliency Toolbox (http://www.saliencytoolbox.net)\n');
fprintf('For licensing details type ''STBlicense'' at the prompt.\n\n');

% need to rename old dll mex files to avoid Matlab warning?
if strcmp(mexext,'mexw32')
  mexfiles = {'mexConv2PreserveEnergy','mexGaussianSubsample','mexLocalMaxima'};
  numMex = length(mexfiles);
  for m = 1:numMex
    bad(m) = (exist([mexfiles{m} '.dll']) == 3);
  end
  if any(bad)
    me = mfilename;
    myPath = which(me);
    myPath = myPath(1:end-(length(me)+2));
    mexPath = [myPath '..' PD 'mex' PD];

    fprintf('This version of Matlab (%s) uses .mexw32 instead of .dll mex files.\n',version);
    for m = find(bad)
      fprintf('Renaming %s.dll to %s.dll.old ... ',mexfiles{m},mexfiles{m});
      [success,message] = movefile([mexPath mexfiles{m} '.dll'],[mexPath mexfiles{m} '.dll.old']);
      if success
        fprintf('done.\n');
      else
        fprintf('failed: %s\n',message);
      end
    end
    fprintf('\n');
    rehash;
  end
end



