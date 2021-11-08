clear all
close all
clc
%% preset directories
baseFolder = pwd;
measuresPath = [baseFolder, '\misure wedge']; 
addpath(genpath(measuresPath));
idxs   = strfind(baseFolder,'\');
addpath(genpath([baseFolder(1:idxs(end)), 'functions']));

%% 1) PREALLOCATE VARIABLES
Fs = 48000; % sampling frequency
duration = 2; % [s]
t = [0:1/Fs:duration - 1/Fs].';
signalLength = duration*Fs;
numberAcquisitions = 5; % number of measurements for each point

%coordinates of the measurement points
measurementPts = [1 1; 1 2; 1 3; 1 4; 1 5;...
                  4 1; 4 2; 4 3; 4 5];
numberPoints = length(measurementPts(:,1)); % number of measurement points in the violin plate
fHigh = 5000;
fLow = 20;
xLengthImg = 400;

%% 2) SUBSTITUTE COMMAS WITH DOTS IN TXT FILES AND CONVERT THEM IN CSV 
% ----------- SKIP IF ALREADY DONE !!!! ------------------------
done = false;
if done == false
 for jj = 1:numberPoints
    fileName = ['14_10_wedge__', int2str(measurementPts(jj,1))...
                 int2str(measurementPts(jj,2)),'_'];
    for ii = 1:numberAcquisitions
        cd([measuresPath,'\txt']);
        tempAcqFilename = [fileName, int2str(ii-1),'.txt'];
        acqFile = fileread(tempAcqFilename);
        fixedAcqFile = strrep(acqFile, ',','.');
        cellAcqFile = textscan(fixedAcqFile, '%f %f %f');
        matAcqFile = [cellAcqFile{1} cellAcqFile{2}];
        cd([measuresPath,'\csv']);
        writeMat2File(matAcqFile,[tempAcqFilename(1:end-4), '.csv'], {'force' 'acc'}, 2, true);
    end
 end
end
%% 3) CALCULATION OF THE H1 ESTIMATOR
plotData = true;

% preallocation
estimatorMatrix = []; % store the H1 estimator
mobilityMatrix = []; % store the mobility transfer function (fft(Y)/fft(X))        
forceTemp = [];
accTemp = [];
FRFTemp = [];
H1Temp  = [];
cohTemp  = [];


% temporary file to compute the frequency responses
nfft = signalLength;
wLength = floor(nfft/10);
window = hamming(wLength);

alpha = 3; % exponential filter decay constant
offsetFilter = 0.001 * Fs; % 100 ms of offset
expFilter = circshift(exp(-alpha*t),offsetFilter);
expFilter(1:offsetFilter) = ones(1,offsetFilter);

for jj = 1:numberPoints
    % fileName with the measurements - general
    fileName = ['14_10_wedge__', int2str(measurementPts(jj,1))...
                 int2str(measurementPts(jj,2)),'_'];
    disp(['extracting FRF: ', fileName(end-4:end-1)]);
    for ii = 1:numberAcquisitions
        disp(['acquisition number ', int2str(ii)]);
        % fileName with the measurements - particular
        tempAcqFilename = [fileName, int2str(ii-1),'.csv'];
        % get data
        rawAccMeasurements = readmatrix(tempAcqFilename);
        forceTemp(:,ii) = rawAccMeasurements(:,1).*expFilter; 
        accTemp(:,ii)   = rawAccMeasurements(:,2).*expFilter;

        % compute the FFT and mobility for each acquisition
        Y1 = FFT(accTemp(:,ii));
        X  = FFT(forceTemp(:,ii));           
        freq = (Fs*(1:(signalLength/2))/signalLength).';      
        FRFTemp(:,ii) =  (1./(1i*2*pi*freq)).*(Y1./X);
        
        [H1, coh, fAxis] = computeH1(forceTemp(:,ii), accTemp(:,ii), nfft, window, Fs, [fLow, fHigh], 'single');
        H1Temp(:,ii) = H1;
        cohTemp(:,ii) = coh;
        if plotData
            figure(2);
            subplot 211
            plot(1:signalLength, forceTemp(:,ii), 1:signalLength, expFilter)
            xlim([-20000 98000])
            subplot 212
            plot(fAxis, db(abs(H1)));
        end
    end
    if jj == 1 
       FRFTemp(:,2) = []; accTemp(:,2) = []; forceTemp(:,2) = [];  
    end
    [H1, coh, fAxis] = computeH1(forceTemp, accTemp, nfft, window, Fs, [fLow, fHigh], 'multiple');
    mobility = mean(FRFTemp(freq<=fAxis(end) & freq >= fAxis(1),:),2);
    
    % plot estimator and coherence
    if plotData
    
    figure(114);  set(gcf, 'Position',  [10, 50, xLengthImg, xLengthImg]);
    subplot 211
     plot(fAxis, db(abs(H1)), 'lineWidth', 1);
     ax = gca; ax.FontSize = 15;
     xlabel('Frequency [Hz]', 'fontSize', 20, 'Interpreter', 'latex');
     ylabel('$|H1_v(f)|$', 'fontSize', 20, 'Interpreter', 'latex');
     xlim([fLow, fHigh])
    subplot 212
     plot(fAxis, coh, 'lineWidth', 1);
     ax = gca; ax.FontSize = 15;
     xlabel('Frequency [Hz]', 'fontSize', 20, 'Interpreter', 'latex');
     ylabel('$Coherence$', 'fontSize', 20, 'Interpreter', 'latex');
    sgtitle(['$H1_{' int2str(measurementPts(jj,1)) int2str(measurementPts(jj,2)), '}$'], 'Interpreter', 'latex');
    xlim([fLow, fHigh])
    
    % plot mobility
    figure(116)
     set(gcf, 'Position',  [650, 50+2/5*xLengthImg, xLengthImg, 2/5*xLengthImg]);
     plot(fAxis, db(abs(mobility)), 'lineWidth', 1);
     ax = gca; ax.FontSize = 15;
     xlabel('Frequency [Hz]', 'fontSize', 16, 'Interpreter', 'latex');
     ylabel('$|Y(f)|$', 'fontSize', 16, 'Interpreter', 'latex');
     title(['$Y_{' int2str(measurementPts(jj,1)) int2str(measurementPts(jj,2)), '}$'], 'Interpreter', 'latex');
    end
    % save results
    estimatorMatrix = [estimatorMatrix H1];
    mobilityMatrix =[ mobilityMatrix   mobility];   
end

if plotData
figure(808)
subplot 211
semilogy(fAxis, (abs(estimatorMatrix)))
title('H1 without SVD')
subplot 212
semilogy(fAxis, (abs(mobilityMatrix)))
title('Y/X')
end



%% IMPORT SIMULATED FRFS
fAxisComsol = 600:2.5:4000;
idxs = find(fAxis >= fAxisComsol(1) & fAxis <= fAxisComsol(end));
vels = [];
xLength = 600;
for jj = 1:numberPoints
    fileName = ['wedge_H' int2str(measurementPts(jj,1))...
                 int2str(measurementPts(jj,2)) '.txt'];
    vel = readTuples(fileName, 1 , false).';
    vels(:,jj) = vel(4:end).';
    dbMesVel = db(abs(mobilityMatrix(idxs, jj))./max(abs(estimatorMatrix(idxs, jj))))
    figure(); box on; clf reset;
    set(gcf, 'Position',  [0, 50, xLength,2/5* xLength]);
    plot(fAxisComsol , db(abs(vels(:,jj))./max(abs(vels(:,jj)))), 'lineWidth', 1.1);
    hold on;
    plot(fAxis(idxs),dbMesVel , 'lineWidth', 1.1); hold off;
    ll = legend('simulated', 'measured');
    set(ll,'Box', 'off');
    set(ll,'Interpreter', 'latex');
    title(fileName(end-6:end-4));
    xlabel('freq [Hz]'); ylabel('|H(f)|'); ax = gca; ax.FontSize = 12;
    ylim([1.1*min(dbMesVel) 15]);
end
