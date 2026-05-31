function T = PSAPResponses(dataDir, outFile)
% PSAPRESPONSES Extract proactive/reactive button proportions from PSAP data.
%
%   T = PSAPRESPONSES() scores every per-participant .mat file in the
%   current folder and returns a table of response proportions.
%
%   T = PSAPRESPONSES(dataDir) scores the .mat files in dataDir.
%
%   T = PSAPRESPONSES(dataDir, outFile) also writes the table to outFile
%   (e.g. 'PSAPResps.xlsx').
%
% Each completed selection is the second row of an {A,B,C}end event in
% InGameVars.Events. Events are split by time into a PROACTIVE phase
% (elapsed < 120 s, before any provocation is possible) and a REACTIVE
% phase (elapsed >= 120 s). The returned proportions are, within each
% phase, the fraction of completed selections that were A, B or C:
%
%   pA, pB, pC  proactive Earn / Steal / Protect proportions
%   rA, rB, rC  reactive  Earn / Steal / Protect proportions
%
% Only files named like a subject ID (P or p followed by digits, e.g.
% P040.mat or p061.mat) are scored; auxiliary files such as *_out.mat are
% ignored, matching the version used for the published analysis.
%
% This is a cleaned port of the original PSAPResponses.m (data/raw/PSAP.zip);
% the phase split and counting logic are unchanged.

PROACTIVE_CUTOFF_S = 120;  % phase boundary (s); see Run_Experiment.m selection()
END_EVENTS = struct('Aend', 'A', 'Bend', 'B', 'Cend', 'C');

if nargin < 1 || isempty(dataDir)
    dataDir = pwd;
end

files = dir(fullfile(dataDir, '*.mat'));
rows = struct('Subject', {}, 'pA', {}, 'pB', {}, 'pC', {}, ...
    'rA', {}, 'rB', {}, 'rC', {});

for k = 1:numel(files)
    [~, name] = fileparts(files(k).name);
    token = regexp(name, '^[Pp](\d+)$', 'tokens', 'once');
    if isempty(token)
        continue  % Skip non-subject files (e.g. P040_out, PSAPResps).
    end
    subject = sprintf('P%03d', str2double(token{1}));

    loaded = load(fullfile(dataDir, files(k).name), 'InGameVars');
    events = loaded.InGameVars.Events;

    proact = struct('A', 0, 'B', 0, 'C', 0);
    react = struct('A', 0, 'B', 0, 'C', 0);
    names = fieldnames(END_EVENTS);
    for j = 1:size(events, 2)
        eventName = events{1, j};
        eventTime = events{2, j};
        if ~ismember(eventName, names)
            continue
        end
        option = END_EVENTS.(eventName);
        if eventTime < PROACTIVE_CUTOFF_S
            proact.(option) = proact.(option) + 1;
        else
            react.(option) = react.(option) + 1;
        end
    end

    nProact = proact.A + proact.B + proact.C;
    nReact = react.A + react.B + react.C;
    rows(end + 1) = struct('Subject', subject, ...
        'pA', proact.A / nProact, 'pB', proact.B / nProact, 'pC', proact.C / nProact, ...
        'rA', react.A / nReact, 'rB', react.B / nReact, 'rC', react.C / nReact); %#ok<AGROW>
end

T = struct2table(rows);

if nargin >= 2 && ~isempty(outFile)
    writetable(T, outFile);
end
end
