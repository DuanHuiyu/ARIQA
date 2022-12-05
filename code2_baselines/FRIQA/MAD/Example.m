return;
%%
% if needed, compile the mexfiles
% mex ical_stat.c
% mex ical_std.c
%% Check the performance:
sizes = 512;
runtime = 1;
% timer = [];
% Feed in integers
ref = round(255*rand(sizes));
dst = round(255*rand(sizes));

for i = 1:runtime
  tic
  [I Map] = MAD_index( ref , dst );
  timer = [timer toc];
  disp(toc);
end
