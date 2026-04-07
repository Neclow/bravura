function [cov_stats, corr_preds] = data_simulation(cohort, nSim)
% Simulation-recovery analysis to assess model identifiability.
%
% For each subject: sample parameters from the prior, simulate data with
% VBA_simulate, re-fit with VBA_NLStateSpaceModel, repeat nSim times.
% Then compute the covariance and correlation of recovered estimates.
%
% Exports to data/cohort_{cohort}/:
%   cov_stats.mat  — [determinant, condition_number] per subject
%   corr_preds.mat — per-subject correlation matrix of recovered parameters
%
% Parameters
% ----------
% cohort : char
%     'a' or 'b'.
% nSim : int, optional
%     Number of simulations per subject (default: 10).

if nargin < 2
    nSim = 10;
end

data_dir = fullfile('data', ['cohort_' cohort]);

%% Load data
file = fullfile(data_dir, 'vba_input.xlsx');
data = importdata(file);
factors = data.data(1:10, :);

if strcmp(cohort, 'b')
    subject_ids = data.textdata(11:end, 1);
else
    subject_ids = data.rowheaders(11:end);
end

nSubject = length(subject_ids);

%% Simulation parameters
n_t = 30;
f_fname = [];
g_fname = @g_Aggression_short;
theta = [];

alpha = 1e1;
sigma = 1e0;
x0 = zeros(1, 1);

%% Run simulation-recovery
cov_stats = zeros(nSubject, 2);
cov_preds = zeros(nSubject, 4, 4);
corr_preds = zeros(nSubject, 4, 4);

for iSubject = 1:nSubject
    fprintf('Subject %d/%d\n', iSubject, nSubject);
    subdata = data.data(10 + iSubject, :);

    options = struct;
    dim = struct('n', 0, 'n_theta', 0, 'n_phi', 0);

    y = subdata;
    options.isYout = isnan(y);
    options.sources.type = 1;
    options.extended = 0;

    prior_mu = 0;
    prior_sigma = 2.30;  % sqrt((1+2^2+4^2+8^2)/4^2)

    [dim, options] = setPriors( ...
        options, dim, 'phi', ...
        'Kr1', prior_mu, prior_sigma, ...
        'Krc', prior_mu, prior_sigma, ...
        'Kp',  prior_mu, prior_sigma, ...
        'Kwc', prior_mu, prior_sigma);

    nTrial = size(subdata, 2);
    dim.n_t = nTrial;

    [u, dim, options] = setInput(options, dim, ...
        'WinSum',      factors(6, :) / max(factors(6, :)), ...
        'ShockedTm1',  factors(8, :), ...
        'ShockedSum',  factors(10, :) / max(factors(10, :)));

    options.DisplayWin = 0;
    options.GnFigs     = 0;
    options.dim        = dim;
    options.verbose    = 0;

    phi_pred = zeros(nSim, 4);
    for iSim = 1:nSim
        % Sample parameters from prior
        phi = normrnd(0, 3.75, size(options.priors.muPhi));

        % Simulate data
        [y_hat, ~, x0, ~, ~, u] = VBA_simulate( ...
            n_t, f_fname, g_fname, theta, phi, u, alpha, sigma, options, x0);

        % Re-fit on simulated data
        [posterior, ~] = VBA_NLStateSpaceModel(y_hat, u, [], g_fname, dim, options);
        phi_pred(iSim, :) = posterior.muPhi;
    end

    % Covariance and correlation of recovered estimates
    cov_pred = cov(phi_pred);
    corr_pred = corr(phi_pred);

    cov_stats(iSubject, :) = [det(cov_pred), cond(cov_pred)];
    cov_preds(iSubject, :, :) = cov_pred;
    corr_preds(iSubject, :, :) = corr_pred;

    fprintf('  cond = %.1f, det = %.4f\n', cond(cov_pred), det(cov_pred));
end

%% Save
save(fullfile(data_dir, 'cov_stats.mat'), 'cov_stats', 'subject_ids');
save(fullfile(data_dir, 'corr_preds.mat'), 'corr_preds', 'cov_preds', 'subject_ids');

fprintf('Simulation-recovery complete. Saved to %s\n', data_dir);

end
