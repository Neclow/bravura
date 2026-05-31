function drawB(w,font_size, color)

[screenXpixels, screenYpixels] = Screen('WindowSize', w);

Screen('TextFont',w, 'Arial');
Screen('TextSize',w, font_size);
Screen('TextStyle', w, 1);

DrawFormattedText(w,'B',screenXpixels * 0.5,screenYpixels * 0.5,color);