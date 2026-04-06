function modelFit_short(file, means, deviations, out_dir)
% Fit the VBA aggression model for all subjects in a data file.
%
% Parameters
% ----------
% file : char
%     Path to vba_input.xlsx (trials as columns, rows 1-10 = design, 11+ = subjects).
% means : double vector
%     Prior means for [Kr1, Krc, Kp, Kwc].
% deviations : double vector
%     Prior standard deviations for [Kr1, Krc, Kp, Kwc].
% out_dir : char
%     Directory to save per-model .mat output.

data = importdata(file);
factors = data.data(1:10, :);

listSubject = data.textdata(11:end, 1);
nSubject = length(listSubject);

summaryResult = zeros(nSubject, 5);

for iSubject = 1:nSubject
    fprintf('Fitting subject %d/%d\n', iSubject, nSubject);
    subdata = data.data(10 + iSubject, :);

    options = struct;
    dim = struct('n', 0, 'n_theta', 0, 'n_phi', 0);

    y = subdata;
    options.isYout = isnan(y);
    y(options.isYout) = 0;
    options.sources.type = 1;
    options.extended = 0;

    [dim, options] = setPriors( ...
        options, dim, 'phi', ...
        'Kr1', means(1), deviations(1), ...
        'Krc', means(2), deviations(2), ...
        'Kp',  means(3), deviations(3), ...
        'Kwc', means(4), deviations(4));

    nTrial = size(subdata, 2);
    dim.n_t = nTrial;

    [u, dim, options] = setInput(options, dim, ...
        'WinSum',      factors(6, :) / max(factors(6, :)), ...
        'ShockedTm1',  factors(8, :), ...
        'ShockedSum',  factors(10, :) / max(factors(10, :)));

    g_fname = @g_Aggression_short;

    options.DisplayWin = 0;
    options.GnFigs     = 0;
    options.dim        = dim;
    options.verbose    = 0;

    [posterior, out] = VBA_NLStateSpaceModel(y, u, [], g_fname, dim, options);
    groupResult.subject(iSubject).posterior = posterior;
    groupResult.subject(iSubject).out = out;
    summaryResult(iSubject, 1:4) = posterior.muPhi';
    summaryResult(iSubject, 5) = out.fit.acc;
end

deviations_str = sprintf('sd%d', deviations);
save(fullfile(out_dir, [deviations_str, '.mat']), 'groupResult', 'summaryResult');
fprintf('Saved %s\n', fullfile(out_dir, [deviations_str, '.mat']));

end
