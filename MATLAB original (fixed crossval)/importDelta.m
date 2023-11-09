function TT = importDelta(filename)

    D=readtable(filename);
    t=D(:,2).Variables;
    t=datetime(t,'ConvertFrom','epochtime','Epoch','1970-01-01','TicksPerSecond',1000);
    TT=array2timetable(D(:,3:end).Variables,'RowTimes',t,'VariableNames',{'Fx','Fy','Fz','Tx','Ty','Tz','Pz'});

end