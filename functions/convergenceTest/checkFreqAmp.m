function [] = checkFreqAmp(f0,fAmps, nPeaks)
%CHECKFREQAMP Summary of this function goes here
%   Detailed explanation goes here
NpeaksAxis = 1:nPeaks;
alphaBetaVar = table2array(readtable('alphaBetaVar.csv'));
names = {'alpha = ' 'beta = '};
alphaBetaNames = cell(size(alphaBetaVar));

for ii = 1:length(alphaBetaVar(:,1))
    alphaBetaNames{ii,1} = [names{1}, num2str(round(alphaBetaVar(ii,1),2))];    
    alphaBetaNames{ii,2} = [names{2}, num2str(round(alphaBetaVar(ii,2),8))];
end

titles = {'alpha variation' 'beta variation'};

    scaledAmps = cell(2,1);
    ampFilenames = {'scaledAmpsAlpha.csv', 'scaledAmpsBeta.csv'};

for ii = 1:2
    eigFreqz = table2array(readtable(['Eigenfrequencies',int2str(ii) ,'.csv']));
    Amps = table2array(readtable(['Amplitudes',int2str(ii) ,'.csv']));
    scaledAmps{ii} = zeros(size(Amps));
    
    figure(ii)
    ax = gca;
    ax.XMinorTick = 'on';
    ax.YMinorTick = 'on';
    ax.TickDir = 'out';
    ax.FontSize = 16;
    title(titles{ii});
    hold on;
    


    for jj = 1:length(Amps(:,1))
    
        ratio = (fAmps(1:5))./abs(Amps(jj, 1:5).');
        gamma = mean(ratio);

        % Frequency/amplitude scaling
        ampsComsol = gamma * abs(Amps(jj, :));
        ampsReal =  fAmps;
        eta = mean(f0(NpeaksAxis)./(ampsReal(NpeaksAxis)) );
        ampsComsol = eta*ampsComsol;
        ampsReal = eta*ampsReal;
        scaledAmps{ii}(jj,:) = ampsComsol;
        
        % Allocate points
        eigenFreqz = eigFreqz(jj, :);
        if jj == 1
            %plot(f0(NpeaksAxis), ampsReal(NpeaksAxis), '-x', 'markerSize' ,5)            
        end
        
        p = plot( eigenFreqz, ampsComsol , '-o', 'markerSize', 5);
        
        hold on;
        mycolor = [0 0.8*jj*1/length(Amps(:,1)) 0];
        set(p,'Color', mycolor);
        
        xlabel('frequency');
        ylabel('amplitude');
        m  = max(eigenFreqz) +20;
        xlim([f0(1)-10, m]);
        ylim([0,1250]);
        legend( alphaBetaNames{1:jj,ii});
    end
    
    ampsTable =  writeMat2File(scaledAmps{ii},ampFilenames{ii}, {'f'}, 1,false);
end

refDiffAlpha = scaledAmps{1}(1,:).* ones(size(scaledAmps{1}));
percVariationAlpha = (scaledAmps{1} - refDiffAlpha)./ (refDiffAlpha(1,:))*100;
stepVariationAlpha = percVariationAlpha - circshift(percVariationAlpha,-1);
stepVariationAlpha(end,:) = zeros(1,length(percVariationAlpha(1,:)));

refDiffBeta = scaledAmps{2}(1,:).* ones(size(scaledAmps{2}));
percVariationBeta = (scaledAmps{2} - refDiffBeta) ./ (refDiffBeta(1,:))*100;
stepVariationBeta = percVariationBeta - circshift(percVariationBeta,-1);
stepVariationBeta(end,:) = zeros(1,length(percVariationBeta(1,:)));


alphaNames = {'alpha' 'f1' 'f2' 'f3' 'f4' 'f5' 'f6' 'f7' 'f8' 'f9' 'f10' 'f11' 'f12' 'f13' 'f14' 'f15'};
betaNames = {'beta' 'f1' 'f2' 'f3' 'f4' 'f5' 'f6' 'f7' 'f8' 'f9' 'f10' 'f11' 'f12' 'f13' 'f14' 'f15'};

alphaPercTable =  writeMat2File([alphaBetaVar(:,1), percVariationAlpha],'alphaPerc.csv', alphaNames, 1,true);
alphaStepTable =  writeMat2File([alphaBetaVar(:,1), stepVariationAlpha],'alphaStepPerc.csv', alphaNames, 1,true);

betaPercTable =  writeMat2File([alphaBetaVar(:,2), percVariationBeta],'betaPerc.csv', betaNames, 1,true);
betaStepTable =  writeMat2File([alphaBetaVar(:,2), stepVariationBeta],'betaStepPerc.csv', betaNames, 1,true);


figure()
hold on;
plot(1:length(eigenFreqz), percVariationAlpha(end,:) , '-o', 'markerSize', 5);
plot(1:length(eigenFreqz), percVariationBeta(end,:) , '-o', 'markerSize', 5);
plot(1:length(eigenFreqz), zeros(size(eigenFreqz)));
xlim([0.8, 15.2]);
xlabel('mode number N');
ylabel(' \Delta [%]');
l = legend('$\alpha = \alpha_r*1.3$', '$\beta = \beta_r*1.3$' );
set(l,'Interpreter','latex');

ax = gca;
    ax.XMinorTick = 'on';
    ax.YMinorTick = 'on';
    ax.TickDir = 'out';
    ax.FontSize = 20;
end