function [dist,i]=findClosestPart(TTpart,TT)
n=size(TT,1);
DIST=[];LEN=[];
for i=1:n/20:n
    [dist,ix,iy]=dtw(TTpart.Variables', TT(1:i,:).Variables');
    DIST=[DIST dist];
    LEN=[LEN i];
    
end
[~,b]=min(DIST);
for i=LEN(max(b-1,1)) : 1 : LEN(min(b+1,length(LEN)))
    [dist,ix,iy]=dtw(TTpart.Variables', TT(1:i,:).Variables');
    DIST=[DIST dist];
    LEN=[LEN i];
    
end
[dist,b]=min(DIST);
i=LEN(b);


end