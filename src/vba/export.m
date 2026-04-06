function export(cohort)
% Export BMA results to CSV and posteriors to .mat for Python.
%
% Exports to data/cohort_{cohort}/:
%   coefficients.csv       — BMA-averaged coefficients [Kr1, Krc, Kp, Kwc]
%   predictions.csv        — VBA predicted P(shock) per trial
%   decisions.csv          — actual binary decisions per trial
%   fit_metrics.csv        — R², accuracy, balanced accuracy, R²_BMA_avg, log evidence
%   subject_ids.csv        — subject ID list
%   free_energy_matrix.csv — per-model free energies
%   vba_posteriors.mat     — muPhi and SigmaPhi per subject (for MC uncertainty)
%
% Parameters
% ----------
% cohort : char
%     'a' or 'b'.

data_dir = fullfile('data', ['cohort_' cohort]);
bma_dir = fullfile(data_dir, 'vba_bma');
model_dir = fullfile(data_dir, 'vba_models');

% ── Load BMA results ────────────────────────────────────────────────────
bma = load(fullfile(bma_dir, 'bma_results.mat'));
posteriors = bma.groupResult.subject.posterior;
nSubjects = length(posteriors);

% ── Load vba_input for subject IDs and decisions ────────────────────────
data = importdata(fullfile(data_dir, 'vba_input.xlsx'));
subject_ids = data.textdata(11:end, 1);
factors = data.data(1:10, :);
nTrials = size(data.data, 2);

% ── Subject IDs ─────────────────────────────────────────────────────────
writetable(table(subject_ids, 'VariableNames', {'subject'}), ...
    fullfile(data_dir, 'subject_ids.csv'));

% ── Coefficients (muPhi) ────────────────────────────────────────────────
coefs = [posteriors.muPhi]';
writetable(array2table(coefs, ...
    'VariableNames', {'Kr1', 'Krc', 'Kp', 'Kwc'}, ...
    'RowNames', subject_ids), ...
    fullfile(data_dir, 'coefficients.csv'), 'WriteRowNames', true);

% ── Posteriors for MC uncertainty ───────────────────────────────────────
mu_all = [posteriors.muPhi]';
sigma_all = zeros(nSubjects, 4, 4);
for s = 1:nSubjects
    sigma_all(s, :, :) = posteriors(s).SigmaPhi;
end
save(fullfile(data_dir, 'vba_posteriors.mat'), 'mu_all', 'sigma_all', 'subject_ids');

% ── Predictions and decisions ───────────────────────────────────────────
win_sum = factors(6, :) / max(factors(6, :));
shocked_tm1 = factors(8, :);
shocked_sum = factors(10, :) / max(factors(10, :));

predictions = zeros(nSubjects, nTrials);
decisions = zeros(nSubjects, nTrials);

for s = 1:nSubjects
    decisions(s, :) = data.data(10 + s, :);
    phi = posteriors(s).muPhi;
    gx = phi(3) + phi(1) * shocked_tm1 + phi(2) * shocked_sum - phi(4) * win_sum;
    predictions(s, :) = 1 ./ (1 + exp(-gx));
end

writematrix(predictions, fullfile(data_dir, 'predictions.csv'));
writematrix(decisions, fullfile(data_dir, 'decisions.csv'));

% ── Fit metrics (all extracted from VBA, no recomputation) ──────────────
files = dir(fullfile(model_dir, '*.mat'));
nFiles = length(files);
R2_all = zeros(nSubjects, nFiles);
acc_all = zeros(nSubjects, nFiles);
bacc_all = zeros(nSubjects, nFiles);
F_all = zeros(nSubjects, nFiles);

for f = 1:nFiles
    loaded = load(fullfile(files(f).folder, files(f).name));
    out = [loaded.groupResult.subject.out];
    fit = [out.fit];
    F_all(:, f) = [out.F]';
    R2_all(:, f) = [fit.R2]';
    acc_all(:, f) = [fit.acc]';
    bacc_all(:, f) = [fit.bacc]';
end

R2_bma_avg = mean(R2_all, 2);
acc_bma_avg = mean(acc_all, 2);
bacc_bma_avg = mean(bacc_all, 2);
log_evidence = sum(F_all, 2);

writetable(array2table( ...
    [R2_bma_avg, acc_bma_avg, bacc_bma, log_evidence], ...
    'VariableNames', {'R2', 'accuracy', 'balanced_accuracy', 'log_evidence'}, ...
    'RowNames', subject_ids), ...
    fullfile(data_dir, 'fit_metrics.csv'), 'WriteRowNames', true);

% ── Free energy matrix ──────────────────────────────────────────────────
writematrix(F_all, fullfile(data_dir, 'free_energy_matrix.csv'));

fprintf('Export complete. Files saved to %s\n', data_dir);

end
