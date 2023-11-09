function plotDtw(x,y,ix,iy)


figure;
offset=30;
arrowIndices=1:100:length(ix);

plot(x);hold on;plot(y+offset);
ax=gca;
xlim = ax.XLim;
ylim = ax.YLim;

for i=arrowIndices
    [x1, y1] = coord2norm(ax, ix(i), x(ix(i)));
    [x2, y2] = coord2norm(ax, iy(i),  y(iy(i))+offset);

   % annotation('arrow',[ix(i) iy(i)]/diff(xlim),[x(ix(i)) y(iy(i))+offset]/diff(ylim));
    annotation('arrow',[x1 x2],[y1,y2]);

end

end