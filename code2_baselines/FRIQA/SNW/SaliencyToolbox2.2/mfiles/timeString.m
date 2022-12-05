% timeString returns the current time and date in a convenient string format.

% This file is part of the SaliencyToolbox - Copyright (C) 2006-2008
% by Dirk B. Walther and the California Institute of Technology.
% See the enclosed LICENSE.TXT document for the license agreement. 
% More information about this project is available at: 
% http://www.saliencytoolbox.net

function s = timeString()

t = clock;
d = date;
s = sprintf('%02d:%02d:%04.1f on %s',t(4),t(5),t(6),d);
