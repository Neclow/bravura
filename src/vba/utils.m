function paths = utils(cohort)
% Return standard VBA directory paths for a cohort, creating them if needed.
%
% Parameters
% ----------
% cohort : char
%     'a' or 'b'.
%
% Returns
% -------
% paths : struct
%     .data_dir   data_v2/cohort_{cohort}
%     .vba_dir    data_v2/cohort_{cohort}/vba
%     .model_dir  data_v2/cohort_{cohort}/vba/models
%     .bma_dir    data_v2/cohort_{cohort}/vba/bma
%     .input_file data_v2/cohort_{cohort}/vba/vba_input.xlsx

paths.data_dir  = fullfile('data_v2', ['cohort_' cohort]);
paths.vba_dir   = fullfile(paths.data_dir, 'vba');
paths.model_dir = fullfile(paths.vba_dir, 'models');
paths.bma_dir   = fullfile(paths.vba_dir, 'bma');

names = fieldnames(paths);
for i = 1:numel(names)
    mkdir(paths.(names{i}));
end

paths.input_file = fullfile(paths.vba_dir, 'vba_input.xlsx');

end
