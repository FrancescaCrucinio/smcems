% Print eps figures without white margins

function printEps(figure, FileName)
% outerpos = axes.OuterPosition;
% ti = axes.TightInset; 
% left = outerpos(1) + ti(1);
% bottom = outerpos(2) + ti(2);
% ax_width = outerpos(3) - ti(1) - ti(3);
% ax_height = outerpos(4) - ti(2) - ti(4);
% axes.Position = [left bottom ax_width ax_height];
figure.PaperPositionMode = 'auto';
fig_pos = figure.PaperPosition;
figure.PaperSize = [fig_pos(3) fig_pos(4)];
print(figure,FileName,'-depsc')
end

