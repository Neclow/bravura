function out=RespirationParameters3(resp, Fs)



% find peaks with proeminence at least as large as the standard deviation
% of the signal, with a width (at half proeminence) of at least 0.5 seconds
% and separated by more than one second (although this one seems redundant
% due to the previous filtering).

%Changed detection of peaks from std to more robust measure of std
RobustStd = (prctile(resp,84.1)-prctile(resp,15.9))/2;

[out.pks,out.locs,out.w,out.p] = findpeaks(resp,Fs,'MinPeakProminence',RobustStd*.8,'MinPeakWidth',0.4,'MinPeakDistance',.8);

%inter breath intervals
out.RR = diff(out.locs);

% cannot remove outliers if numbers are low...
% % outlier replacement for RR time course
% outliers = locateOutliers(locs,RR,'sd',2.5);
% RR=replaceOutliers(locs,RR,outliers,'spline',nan);
% 
% % outlier replacement for respiration amplitude
% outliers = locateOutliers(locs,pks,'sd',2.5);
% pks=replaceOutliers(locs,pks,outliers,'spline',nan);

% calculate ratio: breathing velocity?
out.ratio = out.p./out.w;

% calculate average
out.RRavg = median(out.RR,'omitnan'); %changed to median instead of mean E 08022018
out.Peaksavg = median(out.pks);
out.Wavg = median(out.w,'omitnan');
out.Pavg = median(out.p,'omitnan');
out.Ratioavg = median(out.ratio,'omitnan');

% calculate variability
out.RRvar = std(out.RR);
out.Peaksvar = std(out.pks);
out.Wvar = std(out.w);
out.Pvar = std(out.p);
out.Ratiovar = std(out.ratio);
