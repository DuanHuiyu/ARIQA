% STBgenerateDoc - generates html documentation.
%    This is a wrapper for the m2html program with settings
%    that work for the SaliencyToolbox. You must have m2html
%    in the executable path, and you must change to the 
%    SaliencyToolbox directory before excuting this function.
%
% For m2html see: http://www.artefact.tk/software/matlab/m2html/

% This file is part of the SaliencyToolbox - Copyright (C) 2006-2008
% by Dirk B. Walther and the California Institute of Technology.
% See the enclosed LICENSE.txt document for the license agreement. 
% More information about this project is available at: 
% http://www.saliencytoolbox.net

function STBgenerateDoc

declareGlobal;

if (~ismember(basename(pwd),{'SaliencyToolbox','trunk'}))
  fprintf(['Please change to the SaliencyToolbox base directory and run ' ...
           mfilename ' again.\n']);
  return;
end

htmlDir = ['./doc' PD 'mdoc'];

m2html('mfiles',{'./mfiles/','./mex/'},'htmldir',htmlDir,'recursive','on',...
       'globalHypertextLinks','on','global','on');
