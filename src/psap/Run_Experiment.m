function Run_Experiment()
% POINT SUBTRACTION AGGRESSION PARADIGM (PSAP)
%
% Script author: Joao Rodrigues (2018-05-01).
% Adapted for MATLAB from the Millisecond Inquisit implementation by
% Katja Borchert, Ph.D. (Millisecond Software LLC, 2012; updated 2016).
%
% Based on:
%   Cherek, D.R., Moeller, F.G., Schnapp, W. & Dougherty, D.M. (1997).
%   Studies of Violent and Nonviolent Male Parolees: I. Laboratory and
%   Psychometric Measurements of Aggression. Biological Psychiatry, 42,
%   514-522.
%
% TASK
% A competitive game against a fictitious opponent (in reality the
% computer). The participant earns points (convertible to money) and can
% steal from / protect against the opponent. The participant believes the
% opponent can steal and keep their points; in reality the participant only
% ever sees their own score. Three fixed-ratio (FR) responses are available:
%
%   A: press "A" FRoptionA (100) times  -> earn one point.
%   B: press "B" FRoptionB (10) times   -> steal a point from the opponent
%                                           AND start a protection interval.
%   C: press "C" FRoptionC (10) times   -> start a protection interval.
%
% Once an option is selected no other response is possible until the
% required number of presses is made. When complete, the option vanishes
% for `intermediateduration` before all options become available again.
%
% PROVOCATIONS / PROTECTION
% * Provocations subtract `provocationpoints` (1) whenever no protection is
%   active and the scheduled provocation time has passed. The FIRST
%   provocation is scheduled at 120 s + rand_sample(6, 45); every later one
%   is rescheduled rand_sample(6, 45) s ahead. The 120 s offset means no
%   provocation can occur in the first two minutes -- the boundary used to
%   split a session into proactive (< 120 s) and reactive (>= 120 s) phases
%   in PSAPResponses.m.
% * Protection (started by B or C) lasts rand_sample(45, 90) s, cannot be
%   stacked, and on expiry schedules the next provocation rand_sample(6, 45)
%   s later.
%
% OUTPUT
% Saves `<SubjectID>.mat` containing the `InGameVars` struct. The
% `InGameVars.Events` cell array (rows: name; time in s; running points)
% logs every Start / {A,B,C}start / {A,B,C}end / Steal / Quit event and is
% the input to PSAPResponses.m.
%
% EDITABLE PARAMETERS are grouped in the section below.

BOGUS_IP = '128.178.192.250';  % Fake opponent IP shown during the cover story.
FUNCTIONS_DIR = fullfile(fileparts(mfilename('fullpath')), 'Functions');

close;
sca;
close all;
addpath(genpath(FUNCTIONS_DIR));

%% Subject identification
nsub = input('Insert Subject ID: ', 's');
datafile = [nsub '.mat'];
if exist(datafile, 'file')
    fprintf('\n\n\nA file with this subject ID already exists!\nPlease specify another name!\n\n');
    return
end

%% Event-trigger port
% The computer's parallel port is reserved for shocks; serial is used for
% USB2parallel adapters (which reset to 0 after 10 ms).
prompt = ['Insert COM port for event triggers: \n', ...
    '[Look for Arduino Micro under Ports (COM & LPT) in Device Manager]\n', ...
    'Hint: for initial setup it was COM12\n> '];
COMport = ['\\.\' input(prompt, 's')];

IOPort('CloseAll');
port_type = 'serial';
switch port_type
    case 'parallel'
        ioObj = io32;
        port_obj = ioObj;
        status = io32(ioObj);
    case 'serial'
        com = IOPort('OpenSerialPort', COMport, 'DTR=1');
        port_obj = com;
end
PortStruct.port_type = port_type;
PortStruct.port_obj = port_obj;

%% Bogus network connection (cover story: a remote human opponent)
WaitSecs(0.5);
fprintf(['\n\n\nConnecting to ' BOGUS_IP '...\n']);
WaitSecs(0.5);
h = waitbar(0, ['Connecting to ' BOGUS_IP '.']);
for i = 1:5
    for j = 1:100
        waitbar(j / 100, h);
        WaitSecs(0.01);
    end
end
h = waitbar(1, h, 'Connection established');
WaitSecs(1);
close(h);
WaitSecs(0.5);
fprintf(['\n\nSite ' BOGUS_IP ' is BUSY.\n\n']);
WaitSecs(1);
fprintf('\n\nYour status is: BUSY.\n\n');
WaitSecs(0.5);
fprintf('\n\nPress Enter when ready to start.\n\n');

wait_seconds = 10;
tnow1 = GetSecs;
[~, keyCode, ~] = KbWait([], 0, tnow1 + wait_seconds);
tnow2 = GetSecs;
if sum(keyCode)  % Pressed before timeout: wait out the remaining time.
    fprintf('\n\nYour status is: READY.\n\n');
    remaining_dt = max(0, wait_seconds - (tnow2 - tnow1));
    WaitSecs(remaining_dt);
    beep;
    fprintf(['\n\nSite ' BOGUS_IP ' is now READY!\n\n']);
else  % Timed out: announce the opponent is ready, keep waiting for Enter.
    beep;
    fprintf(['\n\nSite ' BOGUS_IP ' is now READY!\n\n']);
    KbWait;
    fprintf('\n\nYour status is: READY.\n\n');
end
WaitSecs(1);
fprintf('\n\nBoth sides online and READY.\n\n');
WaitSecs(1.5);
fprintf('\n\nExperiment about to start in:\n');
WaitSecs(1);
fprintf('3\n');
WaitSecs(1);
fprintf('2\n');
WaitSecs(1);
fprintf('1\n');

%% Open the experiment screen
PsychDefaultSetup(2);
Screen('Preference', 'SkipSyncTests', 1);
screens = Screen('Screens');
screenNumber = max(screens);
white = WhiteIndex(screenNumber);
[w, rect] = Screen('OpenWindow', max(Screen('screens')), [255 255 255]);

HideCursor;
[screenXpixels, screenYpixels] = Screen('WindowSize', w);
SetMouse(screenXpixels, screenYpixels);

%% ===================== EDITABLE PARAMETERS ============================= %%
Parameters.FRoptionA = 100;       % presses on A to earn a point
Parameters.FRoptionB = 10;        % presses on B to steal + protect
Parameters.FRoptionC = 10;        % presses on C to protect
Parameters.blockduration = 12 * 60;  % duration of one PSAP block (s)
Parameters.number_blocks = 1;     % number of PSAP blocks
Parameters.resttime = 0;          % rest-block duration (min)
Parameters.restperiod_active = 0; % 1 = present a rest block between blocks
Parameters.delay = 0;             % optional built-in response delay (Carre, 2010)
Parameters.intermediateduration = 0.5;  % option-vanish time after completion (s)
Parameters.show_pointcounter_text = 1;  % show text next to the point counter
Parameters.show_pressbutton_text = 1;   % show text next to the press counter
Parameters.restart_total = 0;     % 1 = reset points each block; 0 = carry over
Parameters.rewardpoints = 1;      % points earned per completed A
Parameters.provocationpoints = 1; % points lost per provocation
Parameters.render_time = 0.5;     % screen refresh / key-poll window (s)

%% In-game state
InGameVars.completed = 0;
InGameVars.totalpoints = 0;
InGameVars.presscounts = 0;
InGameVars.protection = 0;
InGameVars.protectcount = 0;
InGameVars.provocationtime = 0;
InGameVars.count_optionA = 0;
InGameVars.count_optionB = 0;
InGameVars.count_optionC = 0;
InGameVars.protectiontime = 0;
InGameVars.poscolortime = 0;
InGameVars.negcolortime = 0;
InGameVars.PSAPstart = 0;
InGameVars.total_A = 0;
InGameVars.total_B = 0;
InGameVars.total_C = 0;
InGameVars.PSAPblocks = 0;
InGameVars.rt = [];
InGameVars.countintermediate = 0;
InGameVars.count_provocations = 0;
InGameVars.instruct_select = 0;
InGameVars.lastPFIstealcount = 0;

%% Display state
script.startTime = GetSecs;
script.nowTime = GetSecs;
script.elapsedtime = script.nowTime - script.startTime;
script.points_size_normal = 30;
script.points_size_big = 50;
script.point_counter_size = 25;
script.letters_size = 50;

script.white = [255 255 255];
script.black = [0 0 0];
script.red = [255 0 0];
script.green = [0 255 0];
script.blue = [0 0 255];

script.Acolor = script.black;
script.Bcolor = script.black;
script.Ccolor = script.black;

script.points_size = script.points_size_normal;
script.points_color = script.black;

script.press_counter_color = script.white;
script.press_counter_size = 50;

script.press_counter_text_color = script.white;
script.press_counter_text_size = 25;

%% Serial-port marker codes
events.startSession = 201;
events.firstPressA = 211;
events.firstPressB = 212;
events.firstPressC = 213;
events.lastPressA = 221;
events.lastPressB = 222;
events.lastPressC = 223;
events.pointStolen = 230;
events.endSession = 202;

InGameVars.Events = {'Start'; script.elapsedtime; 0};  % name; time; points

send_event(PortStruct, events.startSession);

%% Main loop: one selection round per iteration
while script.elapsedtime < Parameters.blockduration

    drawPointsText(w, script.point_counter_size);
    selection();

    out = keyLetterDown(Parameters.render_time);
    if out.pressed
        switch out.letter_pressed
            case 'a'
                InGameVars.Events{1, end + 1} = 'Astart';
                InGameVars.Events{2, end} = script.elapsedtime;
                InGameVars.Events{3, end} = InGameVars.totalpoints;
                send_event(PortStruct, events.firstPressA);
                script.Acolor = script.blue;
                script.Bcolor = script.white;
                script.Ccolor = script.white;
                InGameVars.total_A = InGameVars.total_A + 1;
                optionA_start();
                while InGameVars.count_optionA < Parameters.FRoptionA
                    if optionA()
                        optionA_start();
                    end
                end
                InGameVars.count_optionA = 0;
                InGameVars.totalpoints = InGameVars.totalpoints + Parameters.rewardpoints;
                InGameVars.poscolortime = script.elapsedtime + 1;
                InGameVars.negcolortime = 0;
                script.Acolor = script.white;
                InGameVars.Events{1, end + 1} = 'Aend';
                InGameVars.Events{2, end} = script.elapsedtime;
                InGameVars.Events{3, end} = InGameVars.totalpoints;
                send_event(PortStruct, events.lastPressA);
            case 'b'
                InGameVars.Events{1, end + 1} = 'Bstart';
                InGameVars.Events{2, end} = script.elapsedtime;
                InGameVars.Events{3, end} = InGameVars.totalpoints;
                send_event(PortStruct, events.firstPressB);
                script.Acolor = script.white;
                script.Bcolor = script.blue;
                script.Ccolor = script.white;
                InGameVars.total_B = InGameVars.total_B + 1;
                optionB_start();
                while InGameVars.count_optionB < Parameters.FRoptionB
                    if optionB()
                        optionB_start();
                    end
                end
                % Protection is granted unconditionally (the original
                % "only if provoked since last PFI" guard was disabled).
                InGameVars.protection = 1;
                InGameVars.protectiontime = script.elapsedtime + rand_sample(45, 90);
                InGameVars.lastPFIstealcount = InGameVars.count_provocations;
                InGameVars.protectcount = InGameVars.protectcount + 1;
                InGameVars.count_optionB = 0;
                script.Bcolor = script.white;
                InGameVars.Events{1, end + 1} = 'Bend';
                InGameVars.Events{2, end} = script.elapsedtime;
                InGameVars.Events{3, end} = InGameVars.totalpoints;
                send_event(PortStruct, events.lastPressB);
            case 'c'
                InGameVars.Events{1, end + 1} = 'Cstart';
                InGameVars.Events{2, end} = script.elapsedtime;
                InGameVars.Events{3, end} = InGameVars.totalpoints;
                send_event(PortStruct, events.firstPressC);
                script.Acolor = script.white;
                script.Bcolor = script.white;
                script.Ccolor = script.blue;
                InGameVars.total_C = InGameVars.total_C + 1;
                optionC_start();
                while InGameVars.count_optionC < Parameters.FRoptionC
                    if optionC()
                        optionC_start();
                    end
                end
                % Protection is granted unconditionally (see option B note).
                InGameVars.protection = 1;
                InGameVars.protectiontime = script.elapsedtime + rand_sample(45, 90);
                InGameVars.lastPFIstealcount = InGameVars.count_provocations;
                InGameVars.protectcount = InGameVars.protectcount + 1;
                InGameVars.count_optionC = 0;
                script.Ccolor = script.white;
                InGameVars.Events{1, end + 1} = 'Cend';
                InGameVars.Events{2, end} = script.elapsedtime;
                InGameVars.Events{3, end} = InGameVars.totalpoints;
                send_event(PortStruct, events.lastPressC);
        end
        while InGameVars.countintermediate < 2
            intermediate();
            WaitSecs(Parameters.intermediateduration);
        end
        InGameVars.countintermediate = 0;
        script.Acolor = script.black;
        script.Bcolor = script.black;
        script.Ccolor = script.black;
    end

end
quit();

    function quit()
        InGameVars.Events{1, end + 1} = 'Quit';
        InGameVars.Events{2, end} = script.elapsedtime;
        InGameVars.Events{3, end} = InGameVars.totalpoints;
        send_event(PortStruct, events.endSession);
        save(datafile, 'InGameVars');
        drawTimeOver(w, script.point_counter_size, InGameVars.totalpoints);
        Screen('Flip', w);
        WaitSecs(5);
        KbWait;
        drawEndMessage(w, script.point_counter_size);
        Screen('Flip', w);
        KbWait;
        WaitSecs(5);
        Screen('CloseAll');
        exit;
    end

    function render()
        drawPoints(w, InGameVars.totalpoints, script.points_size, script.points_color);
        drawPointsText(w, script.point_counter_size);
        drawA(w, script.letters_size, script.Acolor);
        drawB(w, script.letters_size, script.Bcolor);
        drawC(w, script.letters_size, script.Ccolor);
        drawPressCounter(w, InGameVars.presscounts, script.press_counter_size, script.press_counter_color);
        drawPressCounterText(w, script.press_counter_text_size, script.press_counter_text_color);
        Screen('Flip', w);
        if script.elapsedtime > Parameters.blockduration
            quit();
        end
    end

    function option_init()
        if InGameVars.negcolortime ~= 0 && script.elapsedtime < InGameVars.negcolortime
            script.points_size = script.points_size_big;
            script.points_color = script.red;
        else
            script.points_size = script.points_size_normal;
            script.points_color = script.black;
        end
        if InGameVars.poscolortime ~= 0 && script.elapsedtime > InGameVars.poscolortime
            InGameVars.poscolortime = 0;
        end
        if InGameVars.negcolortime ~= 0 && script.elapsedtime > InGameVars.negcolortime
            InGameVars.negcolortime = 0;
        end
        if InGameVars.protection == 1 && script.elapsedtime >= InGameVars.protectiontime
            % Protection expired.
            InGameVars.protection = 0;
            InGameVars.protectiontime = 0;
            InGameVars.provocationtime = script.elapsedtime + rand_sample(6, 45);
        end
        if InGameVars.protection == 0 && script.elapsedtime >= InGameVars.provocationtime
            % No protection: a point is stolen.
            InGameVars.totalpoints = InGameVars.totalpoints - Parameters.provocationpoints;
            InGameVars.count_provocations = InGameVars.count_provocations + 1;
            InGameVars.negcolortime = script.elapsedtime + 1;
            InGameVars.poscolortime = 0;
            InGameVars.provocationtime = script.elapsedtime + rand_sample(6, 45);
            script.points_size = script.points_size_big;
            script.points_color = script.red;
            InGameVars.Events{1, end + 1} = 'Steal';
            InGameVars.Events{2, end} = script.elapsedtime;
            InGameVars.Events{3, end} = InGameVars.totalpoints;
            send_event(PortStruct, events.pointStolen);
        end
        script.press_counter_color = script.black;
        script.press_counter_text_color = script.black;
    end

    function optionA_start()
        script.nowTime = GetSecs;
        script.elapsedtime = script.nowTime - script.startTime;
        option_init();
        InGameVars.count_optionA = InGameVars.count_optionA + 1;
        InGameVars.presscounts = InGameVars.count_optionA;
        InGameVars.rt = 0;
        render();
    end

    function resp = optionA()
        script.nowTime = GetSecs;
        script.elapsedtime = script.nowTime - script.startTime;
        option_init();
        render();
        resp = 0;
        out = keyLetterDown(Parameters.render_time);
        if out.pressed && out.letter_pressed == 'a'
            resp = 1;
        end
    end

    function optionB_start()
        script.nowTime = GetSecs;
        script.elapsedtime = script.nowTime - script.startTime;
        option_init();
        InGameVars.count_optionB = InGameVars.count_optionB + 1;
        InGameVars.presscounts = InGameVars.count_optionB;
        InGameVars.rt = 0;
        render();
    end

    function resp = optionB()
        script.nowTime = GetSecs;
        script.elapsedtime = script.nowTime - script.startTime;
        option_init();
        render();
        resp = 0;
        out = keyLetterDown(Parameters.render_time);
        if out.pressed && out.letter_pressed == 'b'
            resp = 1;
        end
    end

    function optionC_start()
        script.nowTime = GetSecs;
        script.elapsedtime = script.nowTime - script.startTime;
        option_init();
        InGameVars.count_optionC = InGameVars.count_optionC + 1;
        InGameVars.presscounts = InGameVars.count_optionC;
        InGameVars.rt = 0;
        render();
    end

    function resp = optionC()
        script.nowTime = GetSecs;
        script.elapsedtime = script.nowTime - script.startTime;
        option_init();
        render();
        resp = 0;
        out = keyLetterDown(Parameters.render_time);
        if out.pressed && out.letter_pressed == 'c'
            resp = 1;
        end
    end

    function intermediate()
        script.nowTime = GetSecs;
        script.elapsedtime = script.nowTime - script.startTime;
        InGameVars.countintermediate = InGameVars.countintermediate + 1;

        if InGameVars.poscolortime ~= 0 && script.elapsedtime < InGameVars.poscolortime
            script.points_size = script.points_size_big;
            script.points_color = script.green;
        end
        if InGameVars.negcolortime ~= 0 && script.elapsedtime < InGameVars.negcolortime
            script.points_size = script.points_size_big;
            script.points_color = script.red;
        end
        if InGameVars.poscolortime ~= 0 && script.elapsedtime > InGameVars.poscolortime
            InGameVars.poscolortime = 0;
        end
        if InGameVars.negcolortime ~= 0 && script.elapsedtime > InGameVars.negcolortime
            InGameVars.negcolortime = 0;
        end
        InGameVars.presscounts = 0;
        if InGameVars.protection == 1 && script.elapsedtime >= InGameVars.protectiontime
            InGameVars.protection = 0;
            InGameVars.protectiontime = 0;
            InGameVars.provocationtime = script.elapsedtime + rand_sample(6, 45);
        end
        if InGameVars.protection == 0 && script.elapsedtime >= InGameVars.provocationtime
            InGameVars.totalpoints = InGameVars.totalpoints - Parameters.provocationpoints;
            InGameVars.count_provocations = InGameVars.count_provocations + 1;
            InGameVars.negcolortime = script.elapsedtime + 1;
            InGameVars.poscolortime = 0;
            InGameVars.provocationtime = script.elapsedtime + rand_sample(6, 45);
            script.points_size = script.points_size_big;
            script.points_color = script.red;
            InGameVars.Events{1, end + 1} = 'Steal';
            InGameVars.Events{2, end} = script.elapsedtime;
            InGameVars.Events{3, end} = InGameVars.totalpoints;
            send_event(PortStruct, events.pointStolen);
        end
        script.press_counter_color = script.white;
        script.press_counter_text_color = script.white;
        render();
    end

    function selection()
        script.nowTime = GetSecs;
        script.elapsedtime = script.nowTime - script.startTime;

        if InGameVars.negcolortime ~= 0 && script.elapsedtime < InGameVars.negcolortime
            script.points_size = script.points_size_big;
            script.points_color = script.red;
        else
            script.points_size = script.points_size_normal;
            script.points_color = script.black;
        end
        if InGameVars.poscolortime ~= 0 && script.elapsedtime > InGameVars.poscolortime
            InGameVars.poscolortime = 0;
        end
        if InGameVars.negcolortime ~= 0 && script.elapsedtime > InGameVars.negcolortime
            InGameVars.negcolortime = 0;
        end
        if InGameVars.protection == 0 && InGameVars.provocationtime == 0
            % Schedule the first provocation >= 120 s in (defines the
            % proactive/reactive phase boundary used by PSAPResponses.m).
            InGameVars.provocationtime = script.elapsedtime + 2 * 60 + rand_sample(6, 45);
        end
        if InGameVars.protection == 1 && script.elapsedtime >= InGameVars.protectiontime
            InGameVars.protection = 0;
            InGameVars.protectiontime = 0;
            InGameVars.provocationtime = script.elapsedtime + rand_sample(6, 45);
        end
        if InGameVars.protection == 0 && script.elapsedtime >= InGameVars.provocationtime
            InGameVars.totalpoints = InGameVars.totalpoints - Parameters.provocationpoints;
            InGameVars.count_provocations = InGameVars.count_provocations + 1;
            InGameVars.negcolortime = script.elapsedtime + 1;
            InGameVars.poscolortime = 0;
            InGameVars.provocationtime = script.elapsedtime + rand_sample(6, 45);
            script.points_size = script.points_size_big;
            script.points_color = script.red;
            InGameVars.Events{1, end + 1} = 'Steal';
            InGameVars.Events{2, end} = script.elapsedtime;
            InGameVars.Events{3, end} = InGameVars.totalpoints;
            send_event(PortStruct, events.pointStolen);
        end
        render();
    end
end

function send_event(s, num)
% Send a one-byte marker on the trigger port, then reset it to 0.
switch s.port_type
    case 'parallel'
        io32(s.port_obj, hex2dec('378'), num);
    case 'serial'
        IOPort('Write', s.port_obj, uint8(num));
end
WaitSecs(0.05);
num = 0;
switch s.port_type
    case 'parallel'
        io32(s.port_obj, hex2dec('378'), num);
    case 'serial'
        IOPort('Write', s.port_obj, uint8(num));
end
end
