function drawPressCounterText(w,font_size,color)

[screenXpixels, screenYpixels] = Screen('WindowSize', w);

Screen('TextFont',w, 'Arial');
Screen('TextSize',w, font_size);
Screen('TextStyle', w, 1);

DrawFormattedText(w,'Bouton presse compteur: ',screenXpixels * 0.25,screenYpixels * 0.75, color);