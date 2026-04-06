function export(cohort)
% Export BMA results to CSV and posteriors to .mat for Python.
%
% Exports to data/cohort_{cohort}/:
%   coefficients.csv    — BMA-averaged coefficients [Kr1, Krc, Kp, Kwc]
%   predictions.csv     — predicted P(shock) per trial
%   decisions.csv       — actual binary decisions per trial
%   fit_metrics.csv     — R², accuracy, balanced accuracy, R²_BMA_avg, log evidence
%   subject_ids.csv     — subject ID list
%   free_energy_matrix.csv — per-model free energies
%   vba_posteriors.mat  — muPhi and SigmaPhi per subject (for MC uncertainty)
%
% Parameters
% ----------
% cohort : char
%     'a' or 'b'.
%
% Example
% -------
%     export('a')

addpath('src/vba');

data_dir = fullfile('data', ['cohort_' cohort]);
bma_dir = fullfile(data_dir, 'vba_bma');
model_dir = fullfile(data_dir, 'vba_models');

% Load BMA results
bma = load(fullfile(bma_dir, 'bma_results.mat'));
posteriors = bma.groupResult.subject.posterior;
nSubjects = length(posteriors);

% Load vba_input for subject IDs and trial data
data = importdata(fullfile(data_dir, 'vba_input.xlsx'));
if strcmp(cohort, 'b')
    subject_ids = data.textdata(11:end, 1);
else
    subject_ids = data.rowheaders(11:end);
end
factors = data.data(1:10, :);

% ── Coefficients (muPhi) ────────────────────────────────────────────────
coefs = [posteriors.muPhi]';  % (nSubjects x 4)
coef_table = array2table(coefs, ...
    'VariableNames', {'Kr1', 'Krc', 'Kp', 'Kwc'}, ...
    'RowNames', subject_ids);
writetable(coef_table, fullfile(data_dir, 'coefficients.csv'), 'WriteRowNames', true);

% ── Posteriors (muPhi + SigmaPhi) for MC uncertainty ────────────────────
mu_all = [posteriors.muPhi]';  % (nSubjects x 4)
sigma_all = zeros(nSubjects, 4, 4);
for s = 1:nSubjects
    sigma_all(s, :, :) = posteriors(s).SigmaPhi;
end
save(fullfile(data_dir, 'vba_posteriors.mat'), 'mu_all', 'sigma_all', 'subject_ids');

% ── Predictions and decisions ───────────────────────────────────────────
predictions = zeros(nSubjects, size(data.data, 2) - 10);
decisions = zeros(nSubjects, size(data.data, 2) - 10);

for s = 1:nSubjects
    subdata = data.data(10 + s, :);
    decisions(s, :) = subdata;

    % Reconstruct predictions from BMA posteriors
    phi = posteriors(s).muPhi;
    nTrial = length(subdata);

    win_sum = factors(6, :) / max(factors(6, :));
    shocked_tm1 = factors(8, :);
    shocked_sum = factors(10, :) / max(factors(10, :));

    gx = phi(3) + phi(1) * shocked_tm1 + phi(2) * shocked_sum - phi(4) * win_sum;
    predictions(s, :) = 1 ./ (1 + exp(-gx));
end

writematrix(predictions, fullfile(data_dir, 'predictions.csv'));
writematrix(decisions, fullfile(data_dir, 'decisions.csv'));

% ── Subject IDs ─────────────────────────────────────────────────────────
id_table = table(subject_ids, 'VariableNames', {'subject'});
writetable(id_table, fullfile(data_dir, 'subject_ids.csv'));

% ── Fit metrics ─────────────────────────────────────────────────────────
% Compute R² per model, then BMA-average
files = dir(fullfile(model_dir, '*.mat'));
nFiles = length(files);
R2_all = zeros(nSubjects, nFiles);
F_all = zeros(nSubjects, nFiles);

for f = 1:nFiles
    loaded = load(fullfile(files(f).folder, files(f).name));
    out = [loaded.groupResult.subject.out];
    F_all(:, f) = out.F;

    % Compute R² per subject for this model
    for s = 1:nSubjects
        ss_res = loaded.groupResult.subject(s).out.fit.ny - ...
                 loaded.groupResult.subject(s).out.suffStat.gx;
        ss_res = sum(ss_res .^ 2);
        y_s = data.data(10 + s, :);
        y_s(isnan(y_s)) = 0;
        ss_tot = sum((y_s - mean(y_s)) .^ 2);
        if ss_tot > 0
            R2_all(s, f) = 1 - ss_res / ss_tot;
        else
            R2_all(s, f) = 0;
        end
    end
end

R2_bma_avg = mean(R2_all, 2);

% Per-subject accuracy from BMA predictions
acc = zeros(nSubjects, 1);
bacc = zeros(nSubjects, 1);
for s = 1:nSubjects
    y_true = decisions(s, :);
    y_pred = predictions(s, :) > 0.5;
    valid = ~isnan(data.data(10 + s, :));
    y_true = y_true(valid);
    y_pred = y_pred(valid);
    acc(s) = mean(y_true == y_pred);

    tp = sum(y_true == 1 & y_pred == 1);
    tn = sum(y_true == 0 & y_pred == 0);
    pos = sum(y_true == 1);
    neg = sum(y_true == 0);
    if pos > 0 && neg > 0
        bacc(s) = 0.5 * (tp / pos + tn / neg);
    else
        bacc(s) = acc(s);
    end
end

% R² from BMA predictions directly
R2_bma = zeros(nSubjects, 1);
for s = 1:nSubjects
    y_s = decisions(s, :);
    valid = ~isnan(data.data(10 + s, :));
    y_s = y_s(valid);
    p_s = predictions(s, valid);
    ss_res = sum((y_s - p_s) .^ 2);
    ss_tot = sum((y_s - mean(y_s)) .^ 2);
    if ss_tot > 0
        R2_bma(s) = 1 - ss_res / ss_tot;
    else
        R2_bma(s) = 0;
    end
end

% Log evidence (sum across models, for BMA weight reference)
log_evidence = sum(F_all, 2);

metrics_table = array2table( ...
    [R2_bma, acc, bacc, R2_bma_avg, log_evidence], ...
    'VariableNames', {'R2', 'accuracy', 'balanced_accuracy', 'R2_BMA_avg', 'log_evidence'}, ...
    'RowNames', subject_ids);
writetable(metrics_table, fullfile(data_dir, 'fit_metrics.csv'), 'WriteRowNames', true);

% ── Free energy matrix ──────────────────────────────────────────────────
writematrix(F_all, fullfile(data_dir, 'free_energy_matrix.csv'));

fprintf('Export complete. Files saved to %s\n', data_dir);

end
