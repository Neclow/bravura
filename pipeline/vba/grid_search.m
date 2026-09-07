function grid_search(cohort)
% Run VBA model fitting across a grid of prior standard deviations.
%
% Parameters
% ----------
% cohort : char
%     'a' or 'b'.
%
% Example
% -------
%     grid_search('a')

addpath('src/vba');

data_dir = fullfile('data', ['cohort_' cohort]);
file = fullfile(data_dir, 'vba_input.xlsx');
out_dir = fullfile(data_dir, 'vba_models');
mkdir(out_dir);

means = [0, 0, 0, 0];
deviations = [1, 2, 4, 8];
n_coefs = length(means);

deviations_grid = unique(nchoosek(repmat(deviations, 1, n_coefs), n_coefs), 'rows');
n_grid = size(deviations_grid, 1);

for i = 1:n_grid
    fprintf('===== Model %d/%d =====\n', i, n_grid);
    modelFit_short(file, means, deviations_grid(i, :), out_dir);
end

fprintf('Done. %d models saved to %s\n', n_grid, out_dir);

end
