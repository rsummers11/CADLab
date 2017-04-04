function h = figure_fullscreen

    scrsz = get(0,'ScreenSize');
    h = figure('Position',[scrsz(1) scrsz(2) scrsz(3) scrsz(4)]); % works on linux
    %h = figure('Position',[scrsz(1) scrsz(4)/10 scrsz(3) (scrsz(4)-2*scrsz(4)/10)]); % exclude task bar (better on windows)
    