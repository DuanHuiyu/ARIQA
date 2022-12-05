% basename strips filename of directory and file extension.
%
% bname = basename(filename) 
%    Removes everything before the right-most occurrence 
%    of the path delimiter PD, and everything after the 
%    left-most occurrence of a dot.

% This file is part of the SaliencyToolbox - Copyright (C) 2006-2008
% by Dirk B. Walther and the California Institute of Technology.
% See the enclosed LICENSE.TXT document for the license agreement. 
% More information about this project is available at: 
% http://www.saliencytoolbox.net

function bname = basename(filename)

declareGlobal;

slash = find(filename == PD);
if isempty(slash)
  left = 1;
else
  left = slash(end)+1;
end

dot = find(filename == '.');
if isempty(dot)
  right = length(filename);
else
  right = dot(end)-1;
end

if (left > right)
  bname = filename(left:end);
else
  bname = filename(left:right);
end

