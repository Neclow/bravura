function drawTime(w,elapsedtime,font_size,color_vect)

[screenXpixels, screenYpixels] = Screen('WindowSize', w);

Screen('TextFont',w, 'Arial');
Screen('TextSize',w, font_size);
Screen('TextStyle', w, 1);

DrawFormattedText(w,['Temps écoulé (s): ' num2str(round(elapsedtime))],screenXpixels * 0.1,screenYpixels * 0.1,color_vect);