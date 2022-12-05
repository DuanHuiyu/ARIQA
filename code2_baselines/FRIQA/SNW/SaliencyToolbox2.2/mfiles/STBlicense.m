% STBlicense - displays the SaliencyToolbox license agreement.

% This file is part of the SaliencyToolbox - Copyright (C) 2006-2008
% by Dirk B. Walther and the California Institute of Technology.
% See the enclosed LICENSE.TXT document for the license agreement. 
% More information about this project is available at: 
% http://www.saliencytoolbox.net

function STBlicense
me = mfilename;
myPath = which(me);
myPath = myPath(1:end-(length(me)+2));
licenseFile = [myPath '../LICENSE.TXT'];
more on;
try
  type(licenseFile);
catch
  % do nothing
end
more off;
