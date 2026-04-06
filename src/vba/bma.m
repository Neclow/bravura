function bma(cohort)
% Run Bayesian Model Averaging over grid search results.
%
% Parameters
% ----------
% cohort : char
%     'a' or 'b'.
%
% Example
% -------
%     bma('a')

data_dir = fullfile('data', ['cohort_' cohort]);
model_dir = fullfile(data_dir, 'vba_models');
out_dir = fullfile(data_dir, 'vba_bma');
mkdir(out_dir);

files = dir(fullfile(model_dir, '*.mat'));
nFiles = length(files);
fprintf('Found %d models in %s\n', nFiles, model_dir);

% Read subject count from first file
tmp = load(fullfile(files(1).folder, files(1).name));
nSubjects = length(tmp.groupResult.subject);
fprintf('Subjects: %d\n', nSubjects);

% Collect posteriors and free energies across models
P = struct();
F = zeros(nSubjects, nFiles);

for f = 1:nFiles
    path = fullfile(files(f).folder, files(f).name);
    loaded = load(path);
    if f == 1
        P = [loaded.groupResult.subject.posterior];
    else
        P(f, :) = [loaded.groupResult.subject.posterior];
    end
    out = [loaded.groupResult.subject.out];
    F(:, f) = out.F;
    fprintf('Loaded model %d/%d: %s\n', f, nFiles, files(f).name);
end

% Run BMA per subject
P_BMA = struct();
for s = 1:nSubjects
    if s == 1
        P_BMA = VBA_BMA(P(:, s), F(s, :)');
    else
        P_BMA(s, :) = VBA_BMA(P(:, s), F(s, :)');
    end
    if mod(s, 20) == 0
        fprintf('BMA subject %d/%d\n', s, nSubjects);
    end
end

% Save BMA results
groupResult = struct();
groupResult.subject = struct();
groupResult.subject.posterior = P_BMA';
summaryResult = [P_BMA.muPhi]';

save(fullfile(out_dir, 'bma_results.mat'), 'groupResult', 'summaryResult');

% Also save free energy matrix for reference
save(fullfile(out_dir, 'free_energy.mat'), 'F');

fprintf('BMA complete. Results saved to %s\n', out_dir);

end
