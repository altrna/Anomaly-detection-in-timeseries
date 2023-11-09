clear all;close all;
verbose = 1;

TT1=importDelta('TireAssemblyFT_1.csv');
TT2=importDelta('TireAssemblyFT_2.csv');
TT3=importDelta('TireAssemblyFT_3.csv');
TT4=importDelta('TireAssemblyFT_4.csv');
TT={TT1,TT2,TT3,TT4};


TR=TT([2:4]);

TRall=[];for i=1:length(TR),TRall=[TRall;TR{i}.Variables];end
mn=mean(TRall);st=std(TRall);
for i=1:length(TR),TR{i}.Variables=(TR{i}.Variables-mn)./repmat(st,size(TR{i},1),1);end
TS=TT{1};
TS.Variables=(TS.Variables-mn)./repmat(st,size(TS,1),1);

n=size(TS,1);
DIST=[];I=[];
for step=100:50:n

    TSpart = TS(1:step,:);
    
    [~,iopt] = findClosestPart(TSpart,TR{1});
    
    TT1 = TSpart;TT2 = TR{1}(1:iopt,:);
    
    [dist,ix,iy]=dtw(TT1.Variables', TT2.Variables');
    
    %DIST = [DIST dist/length(ix)];
    DIST = [DIST  sum(sqrt(sum((TT1(ix(end),:).Variables-TT2(iy(end),:).Variables).^2,2)))];
    I = [I iopt];    
    
    if verbose
        close all;figure('Units','normalized','Position',[0 0 1 1])
        
        for step1=1:7
        
        subplot(7,2,2*(step1-1)+1);
        hold on;
        plot(TT1.Time-TT1.Time(1),TT1(:,step1).Variables);
   %    plot(TT2.Time-TT2.Time(1),TT2(:,step).Variables);
        
        subplot(7,2,2*(step1-1)+2);
        hold on;
        plot(TT1(ix,step1).Variables);
        plot(TT2(iy,step1).Variables)
        
        end
        subplot(7,2,1);ylabel('Force X');
        subplot(7,2,3);ylabel('Force Y');
        subplot(7,2,5);ylabel('Force Z');
        subplot(7,2,7);ylabel('Torque X');
        subplot(7,2,9);ylabel('Torque Y');
        subplot(7,2,11);ylabel('Torque Z');
        subplot(7,2,13);ylabel('Position Z');
        subplot(7,2,13);xlabel('Time');
        subplot(7,2,14);xlabel('Samples');
        
        drawnow;
    end
    disp(step);
end
