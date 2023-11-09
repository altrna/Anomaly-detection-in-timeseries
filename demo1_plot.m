clear all;close all;
TT1=importDelta('TireAssemblyFT_1.csv');
TT2=importDelta('TireAssemblyFT_2.csv');
TT3=importDelta('TireAssemblyFT_3.csv');
TT4=importDelta('TireAssemblyFT_4.csv');
TT={TT1,TT2,TT3,TT4};

figure
for i=1:1
    
    for j=1:4
    subplot(4,1,j);
    hold on;
        plot(TT{j}.Time-TT{j}.Time(1),TT{j}(:,i).Variables);

    end
end

 
%figure;dtw(TT1.Fx,TT2.Fx);
figure;dtw(TT2.Fz,TT3.Fz);
%figure;dtw(TT1.Fx,TT4.Fx);