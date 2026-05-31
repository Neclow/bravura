function drawPointsText(w,font_size)

[screenXpixels, screenYpixels] = Screen('WindowSize', w);

Screen('TextFont',w, 'Arial');
Screen('TextSize',w, font_size);
Screen('TextStyle', w, 1);

DrawFormattedText(w,'Points Totaux: ',screenXpixels * 0.25,screenYpixels * 0.2,[0 0 0]);