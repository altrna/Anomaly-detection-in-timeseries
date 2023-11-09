clear all;close all;
TT1=importDelta('../data/TireAssemblyFT_2.csv');
TT2=importDelta('../data/TireAssemblyFT_3.csv');

[dist,ix,iy]=dtw(TT1.Variables', TT2.Variables')

figure
for i=1:7
    
    subplot(7,2,2*(i-1)+1);
    hold on;
    plot(TT1.Time-TT1.Time(1),TT1(:,i).Variables);
    plot(TT2.Time-TT2.Time(1),TT2(:,i).Variables);

    subplot(7,2,2*(i-1)+2);
    hold on;
    plot(TT1(ix,i).Variables);
    plot(TT2(iy,i).Variables)

    end


