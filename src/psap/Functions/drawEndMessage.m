function drawEndMessage(w,font_size)

[screenXpixels, screenYpixels] = Screen('WindowSize', w);

Screen('TextFont',w, 'Arial');
Screen('TextSize',w, font_size);
Screen('TextStyle', w, 1);

DrawFormattedText(w,'Merci d''avoir participé. Appuyez sur ''Enter'' pour quitter.',screenXpixels * 0.3,screenYpixels * 0.5,[0 0 0]);