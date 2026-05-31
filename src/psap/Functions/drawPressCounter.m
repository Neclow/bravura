function drawPressCounter(w,press_counts,font_size,color)

[screenXpixels, screenYpixels] = Screen('WindowSize', w);

Screen('TextFont',w, 'Arial');
Screen('TextSize',w, font_size);
Screen('TextStyle', w, 1);

DrawFormattedText(w,num2str(press_counts),screenXpixels * 0.5,screenYpixels * 0.75, color);