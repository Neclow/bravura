function drawPoints(w,points,font_size,color_vect)

[screenXpixels, screenYpixels] = Screen('WindowSize', w);

Screen('TextFont',w, 'Arial');
Screen('TextSize',w, font_size);
Screen('TextStyle', w, 1);

DrawFormattedText(w,num2str(points),screenXpixels * 0.5,screenYpixels * 0.2,color_vect);