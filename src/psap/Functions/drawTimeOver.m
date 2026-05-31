function drawTimeOver(w,font_size,points)

[screenXpixels, screenYpixels] = Screen('WindowSize', w);

Screen('TextFont',w, 'Arial');
Screen('TextSize',w, font_size);
Screen('TextStyle', w, 1);

DrawFormattedText(w,['Votre temps est fini. Vous avez gagné ' num2str(points) ' points.'],screenXpixels * 0.3,screenYpixels * 0.5,[0 0 0]);