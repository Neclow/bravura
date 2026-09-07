function outT = stress_aggression_NF(mode)

if nargin < 1, mode = 'valid'; end
assert(ismember(mode, {'valid', 'legacy'}), 'mode must be ''valid'' or ''legacy''');

script_dir = fileparts(mfilename('fullpath'));
src_physio = fullfile(script_dir, '..', '..', 'src', 'physio');
if exist(src_physio, 'dir')
    addpath(src_physio);
else
    warning('src/physio not found at %s — add it to the MATLAB path manually.', src_physio);
end

ext_dir = fullfile(script_dir, '..', '..', 'extern');
physionet_dir = fullfile(ext_dir, 'PhysioNet-Cardiovascular-Signal-Toolbox');
vollmer_dir = fullfile(ext_dir, 'MarcusVollmer-HRV');
ledalab_dir = fullfile(ext_dir, 'ledalab');
ledalab_batch = fullfile(ledalab_dir, 'BatchData');

data_raw = fullfile(script_dir, '..', '..', 'data', 'raw');
prev_dir = pwd;
cd(data_raw);

Fs = 1000;
dsample=4;
dsample_resp=10;
Fs_ds = 100; % Fs from the downsampled timeseries imported...

files=dir(fullfile('Sync_phys','BF*.mat'));

h1 = waitbar(0,'Please wait...');
s1 = clock;
start_num=1;
end_num=length(files);
j=0;
for i=1:length(files)
    
    phys_file_name = ['Sync_phys/' files(i).name];
    events_file_name = ['MatlabEvents/' files(i).name];
    
    [filepath,fname,~] = fileparts(phys_file_name);
    split=strsplit(fname,'_');
    name=split{1};
    
    j=j+1
    
    load(phys_file_name)
    load(events_file_name)
    auxS(j).Subject = name;
    try
        event_times = check_events(event, mode);
    catch ME
        warning(['Problem in subject ', name, ME.message])
        continue;
    end
    
    
    
    fn = fieldnames(event_times);
    h2 = waitbar(0,'Please wait...');
    s2 = clock;
    for k=1:numel(fn)
        tic
        % event_times.(fn{k})
        tr= event_times.(fn{k});
        
        % ECG
        addpath(genpath(physionet_dir));
        try
            HRVout = My_Main_HRV_Analysis(TT_ECG(tr,:).ECG,[],'ECGWaveform',My_InitializeHRVparams('StressTest1000'));
        catch ME
            rmpath(genpath(physionet_dir));
            warning(['Problem in subject ', name, ME.message])
            continue;
        end
        rmpath(genpath(physionet_dir));
        
        
        addpath(genpath(vollmer_dir));
        % Computation of local HRV measures
        RR_loc = HRVout.NN;

        rrHRV = HRV.rrHRV(RR_loc);

        HR    = HRV.HR(RR_loc);
        [TRI,TINN] = HRV.triangular_val(RR_loc);
        rmpath(genpath(vollmer_dir));
        
        auxS(j).(['HR_' fn{k}]) = HR;
        auxS(j).([HRVout.HRVtitle{10} '_' fn{k}]) = HRVout.HRVout(10);
        auxS(j).([HRVout.HRVtitle{11} '_' fn{k}]) = HRVout.HRVout(11);
        auxS(j).([HRVout.HRVtitle{12} '_' fn{k}]) = HRVout.HRVout(12);
        auxS(j).([HRVout.HRVtitle{16} '_' fn{k}]) = HRVout.HRVout(16);
        auxS(j).([HRVout.HRVtitle{17} '_' fn{k}]) = HRVout.HRVout(17);
        auxS(j).([HRVout.HRVtitle{18} '_' fn{k}]) = HRVout.HRVout(18);
        auxS(j).([HRVout.HRVtitle{19} '_' fn{k}]) = HRVout.HRVout(19);
        auxS(j).([HRVout.HRVtitle{20} '_' fn{k}]) = HRVout.HRVout(20);
        auxS(j).([HRVout.HRVtitle{21} '_' fn{k}]) = HRVout.HRVout(21);
        auxS(j).([HRVout.HRVtitle{23} '_' fn{k}]) = HRVout.HRVout(23);
        auxS(j).([HRVout.HRVtitle{24} '_' fn{k}]) = HRVout.HRVout(24);
        auxS(j).([HRVout.HRVtitle{25} '_' fn{k}]) = HRVout.HRVout(25);
        auxS(j).([HRVout.HRVtitle{26} '_' fn{k}]) = HRVout.HRVout(26);
        auxS(j).([HRVout.HRVtitle{27} '_' fn{k}]) = HRVout.HRVout(27);
        auxS(j).([HRVout.HRVtitle{28} '_' fn{k}]) = HRVout.HRVout(28);
        auxS(j).([HRVout.HRVtitle{29} '_' fn{k}]) = HRVout.HRVout(29);
        auxS(j).(['TRI_' fn{k}]) = TRI;
        auxS(j).(['TINN_' fn{k}]) = TINN;
        auxS(j).(['rrHRV_' fn{k}]) = rrHRV;
        
        % breathing
        outputResp=RespirationParameters3(TT_Resp(tr,:).Resp_ds, Fs_ds);
        auxS(j).(['RespRate_' fn{k}]) = 60/outputResp.RRavg;
        auxS(j).(['RespWidth_' fn{k}]) = outputResp.Wavg;
        auxS(j).(['RespDepth_' fn{k}]) = outputResp.Pavg;
        auxS(j).(['RespRatio_' fn{k}]) = outputResp.Ratioavg;
        
        % EDA (requires ledalab)
        if exist(ledalab_dir, 'dir')
            addpath(genpath(ledalab_dir));
            eda_batch = fullfile(pwd, '_eda_tmp');
            if ~exist(eda_batch, 'dir'), mkdir(eda_batch); end
            current_folder = pwd;
            time_step_eda = 1/Fs_ds;
            data.conductance = TT_EDA(tr,:).EDA_ds';
            data.time = [0:length(TT_EDA(tr,:).EDA_ds)-1]*time_step_eda;
            data.event(1).time=0;
            data.event(1).nid=1;
            data.event(1).name=fn{k};
            save(fullfile(eda_batch, 'leda_temp_EDA.mat'), 'data')

            Ledalab([eda_batch filesep], 'open', 'mat', 'filter',[4 1], 'smooth', {'adapt'},'downsample', 2 ,'analyze','CDA', 'optimize',2, 'export_era', [0 data.time(end) 0.05 1])
            cd(current_folder)
            load(fullfile(current_folder, 'leda_temp_EDA_era.mat'))
            delete(fullfile(current_folder, 'batchmode_protocol.mat'))
            delete(fullfile(eda_batch, 'leda_temp_EDA.mat'))
            delete(fullfile(current_folder, 'leda_temp_EDA_era.mat'))
            toc
            nSCRcda = results.CDA.nSCR;
            Toniccda = results.CDA.Tonic;
            AmpSumcda = results.CDA.Tonic;
            rmpath(genpath(ledalab_dir));
            auxS(j).(['nSCRcda_' fn{k}]) = nSCRcda/(data.time(end)/60);
            auxS(j).(['Toniccda_' fn{k}]) = Toniccda;
            auxS(j).(['AmpSumcda_' fn{k}]) = AmpSumcda/nSCRcda;

            nSCRttp = results.TTP.nSCR;
            AmpSumttp = results.TTP.AmpSum;
            auxS(j).(['nSCRttp_' fn{k}]) = nSCRttp/(data.time(end)/60);
            auxS(j).(['AmpSumttp_' fn{k}]) = AmpSumttp/nSCRttp;
        else
            if k == 1 && j == 1
                warning('ledalab-349 not found in %s — skipping EDA analysis.', ext_dir);
            end
        end
        
        %         if k ==1
        %             is = etime(clock,s2);
        %             esttime2 = is * numel(fn);
        %         end
        is = etime(clock,s2);
        tremaining = is/k * (numel(fn)-k);
        if isvalid(h2)
            h2 = waitbar(k/numel(fn),h2,...
                ['Periods remaining time =',num2str(tremaining,'%4.1f'),'sec' ]);
        end

    end
    if isvalid(h2), close(h2); end
    %     if i ==start_num
    %         is = etime(clock,s1);
    %         esttime1 = is * (end_num-start_num+1);
    %     end
    is = etime(clock,s1);
    tremaining = is/(i-start_num+1) * (end_num-i); %average time per subject * remaining subjects.
    if isvalid(h1)
        h1 = waitbar((i-start_num+1)/(end_num-start_num+1),h1,...
            ['Subjects remaining time =',num2str(tremaining/60,'%4.1f'),'minutes' ]);
    end
end
outT= struct2table(auxS);
if isvalid(h1), close(h1); end
cd(prev_dir)

function out = check_events(event, mode)

if nargin < 2, mode = 'valid'; end

nids = [event.nid];
test_start_id = 101;
choice_id = 107;
pos = find(nids == test_start_id);

if isempty(pos)
    error('Aggression tests not existent')
end

if length(pos) > 2
    if strcmp(mode, 'legacy')
        warning('more than two Aggression tests. Taking the last 2.')
        pos = pos(end-1:end);
    else
        % Find the two T101s with exactly 15 T107 (choice) events each
        boundaries = [pos, length(nids) + 1];
        valid = [];
        for vi = 1:length(pos)
            block_nids = nids(pos(vi):boundaries(vi+1)-1);
            if sum(block_nids == choice_id) == 15
                valid(end+1) = vi; %#ok<AGROW>
            end
        end
        if length(valid) >= 2
            pos = pos([valid(1), valid(end)]);
        else
            warning('Could not find two valid 15-trial blocks. Falling back to first and last T101.')
            pos = pos([1, end]);
        end
    end
end

opponent_1 = pos(1);
opponent_2 = pos(2);

test_end_id = 104;
pos = find(nids == test_end_id);
if isempty(pos)
    error('Aggression tests end marker not existent')
elseif length(pos) > 2
    pos = pos([1, end]);
end
end_aggression = pos(end);


% 2.5 min before aggression
tr = timerange(seconds(event(opponent_1).seconds)-minutes(2.5),seconds(event(opponent_1).seconds)-minutes(0));
out.Pre = tr;
% 0 to 2.5 min after aggression opponent 1
tr = timerange(seconds(event(opponent_1).seconds)-minutes(0),seconds(event(opponent_1).seconds)+minutes(2.5));
out.Op1T1 = tr;
% 2.5 to 5 min after aggression opponent 1
tr = timerange(seconds(event(opponent_1).seconds)+minutes(2.5),seconds(event(opponent_1).seconds)+minutes(5));
out.Op1T2 = tr;

% 0 to 2.5 min after aggression opponent 2
tr = timerange(seconds(event(opponent_2).seconds)+minutes(0),seconds(event(opponent_2).seconds)+minutes(2.5));
out.Op2T1 = tr;
% 2.5 to 5 min after aggression opponent 2
tr = timerange(seconds(event(opponent_2).seconds)+minutes(2.5),seconds(event(opponent_2).seconds)+minutes(5));
out.Op2T2 = tr;
% 2.5 min after aggression opponent 2
tr = timerange(seconds(event(end_aggression).seconds)+minutes(0),seconds(event(end_aggression).seconds)+minutes(2.5));
out.Post = tr;


