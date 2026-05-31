function drawProvocationTime(w,provocationtime,font_size,color_vect)

[screenXpixels, screenYpixels] = Screen('WindowSize', w);

Screen('TextFont',w, 'Arial');
Screen('TextSize',w, font_size);
Screen('TextStyle', w, 1);

DrawFormattedText(w,[num2str(provocationtime) ' seconds'],screenXpixels * 0.9,screenYpixels * 0.1,color_vect);