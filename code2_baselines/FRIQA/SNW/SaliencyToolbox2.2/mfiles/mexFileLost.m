% mexFileLost - displays a message if a mex file couldn't be found.

% This file is part of the SaliencyToolbox - Copyright (C) 2006-2008
% by Dirk B. Walther and the California Institute of Technology.
% See the enclosed LICENSE.TXT document for the license agreement. 
% More information about this project is available at: 
% http://www.saliencytoolbox.net

function mexFileLost

name = [callingFunctionName '.' mexext];

fprintf('\nThis is only the help file for the mex file %s.\n',name);
fprintf('Make sure that the SaliencyToolbox/mex directory is in your Matlab path.\n');
fprintf('If you still see this message, then %s might need\n',name);
fprintf('to be compiled for your system. The source code is in:\n');
fprintf('SaliencyToolbox/src/%s.cpp\n',name);
fprintf('See SaliencyToolbox/doc/index.html for instructions on how\n');
fprintf('to compile the code.\n\n');

error(['MEX file ' name ' could not be found.']);
