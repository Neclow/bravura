function out = keyLetterDown_practice(duration)
% actually should be calle key letter up..
%KbStrokeWait
out.tNow = GetSecs;

% end wait
%[out.secs, out.keyCode, out.deltaSecs] =altKbWait([],0,duration,out.tNow);
%[out.secs, out.keyCode, out.deltaSecs] =KbStrokeWait([], out.tNow+duration);
[out.secs, out.keyCode, out.deltaSecs] =KbWait([], 2, out.tNow+duration);
out.dT=out.secs-out.tNow;
% Wait for a minimum time where responses would be too fast o be legit
WaitSecs(0.12);

yesKey=find(out.keyCode==1);
actual_key=KbName(yesKey);
if isempty(actual_key)
    out.letter_pressed=[];
    out.pressed=0;
    return;
end
try
    out.letter_pressed=actual_key(1);
    if strcmp(out.letter_pressed,'a') || strcmp(out.letter_pressed,'b') || strcmp(out.letter_pressed,'c') || strcmp(out.letter_pressed,'s')|| strcmp(out.letter_pressed,'q')
        out.pressed=1;
    else
        out.pressed=0;
    end
catch
    out.pressed=0;
    return;
end
