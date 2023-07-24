%%  ______________________ MAIN COMPUTE EXP FRF ___________________
% THIS PROGRAM COMPUTES THE EXPERIMENTAL FRFS WITH THE H1 ESTIMATOR
% STARTING FROM FORCE AND ACCELERATION MESUREMENTS 
% A) computes H1 estimator
% B) saves it
% -------------------------------------------------------------------------
% summary:
% section 0) PRESET DIRECTORIES, INIT VARIABLES
% section 1) CALCULATION OF THE H1 ESTIMATOR
% section 2) PEAK ANALYSIS AND SAVE FRFs
% -------------------------------------------------------------------------

%% SECTION 0) PRESET DIRECTORIES, INIT VARIABLES
% -------------------------------------------------------------------------
% A) directories
% close all images, clear all variables, clear command window
clear = 0; close = 0;
if clear, clear all; end
if close, close all; end
clc

baseFolder = pwd;
measuresPath = [baseFolder, '\FRF_data']; 
addpath(genpath(measuresPath));
idxs   = strfind(baseFolder,'\');
addpath(genpath([baseFolder(1:idxs(end)), 'functions']));

% B) set variables
measuredSamples = {'1' '2' '3' '4' '5' '6' '7' '8' '9' '10'}; % plates labels
nPlates = length(measuredSamples); % number of measurement points in the violin plate
fHigh = 2000;     % high bound of the frequency axis
fLow = 20;        % lower bound of the frequency axis
xLengthImg = 800; % x length of the image 

%% section 1) CALCULATION OF THE H1 ESTIMATOR
% -------------------------------------------------------------------------
% boolean to plot a figure or not
plotData = 0;

% preallocation
H1_matrix = []; % store the H1 estimator
coh_matrix = []; % store the coherence of the estimators
force_matrix = [];    
acc_matrix = [];

% set variables
Fs = 48000;   % sampling frequency
duration = 2; % duration of the signals [s]
t = [0:1/Fs:duration - 1/Fs].'; % time axis from 0 to duration with step 1/Fs
signalLength = duration*Fs;     % signal length
numberAcquisitions = 5;         % number of measurements for each point

% set FFT variables - i.e. signal length and window 
nfft = signalLength;
wLength = floor(nfft/2);
window =(ones(1  ,wLength));

% make an exponential filter to clean force and acceleration signals
exFilterWeights = 1; % exponential filter decay constant
offsetFilter = round(0 * Fs); % 100 ms of offset
expFilter = circshift(exp(-exFilterWeights*t),offsetFilter);
expFilter(1:offsetFilter) = ones(1,offsetFilter);

for plateN = 1:nPlates
    % fileName with the measurements - general
    disp(['extracting FRF of plate ', measuredSamples{plateN}]);
    for acqN = 1:numberAcquisitions
        disp(['acquisition number ', int2str(acqN)]);
        % fileName with the measurements - particular
        fileName = ['plate_' num2str(plateN) '_____15_11_21_____frf_' num2str(acqN) '.csv'];
        % get data
        rawAccMeasurements = readmatrix(fileName);
        force_matrix(:,acqN) = rawAccMeasurements(:,1).*expFilter; 
        acc_matrix(:,acqN)   = rawAccMeasurements(:,2).*expFilter;
    end
    % compute the H1 estimator of the mobility (velocity/force) from force
    % and acceleration
    [H1, coh, fAxis] = computeH1(force_matrix, acc_matrix, nfft, window, Fs, [fLow, fHigh], 'multiple');
    
    % plot estimator and coherence
    if plotData   
        figure(plateN);  set(gcf, 'Position',  [5, 50, xLengthImg, 3/5*xLengthImg]);
        subplot 211
         plot(fAxis, db(abs(H1)), 'lineWidth', 1);
         ax = gca; ax.FontSize = 15;
         xlabel('Frequency [Hz]', 'fontSize', 20, 'Interpreter', 'latex');
         ylabel('$|H1_v(f)|$', 'fontSize', 20, 'Interpreter', 'latex');
         xlim([fLow, 500])
        subplot 212
         plot(fAxis, coh, 'lineWidth', 1);
         ax = gca; ax.FontSize = 15;
         xlabel('Frequency [Hz]', 'fontSize', 20, 'Interpreter', 'latex');
         ylabel('$Coherence$', 'fontSize', 20, 'Interpreter', 'latex');
        sgtitle(['$H1_{' measuredSamples{plateN}, '}$'], 'Interpreter', 'latex');
        xlim([fLow, 500])
    end
    
    H1_matrix = [H1_matrix H1];
    coh_matrix = [coh_matrix coh(:)];
end

if plotData
figure(808)
plot(fAxis, db(abs(H1_matrix)))
title('H1 ')
end


%% section 2) PEAK ANALYSIS AND SAVE FRFs
% -------------------------------------------------------------------------
% close all
% variables
fMin =20;
fMax = 2000;
minPkWidths = [3  2.7    2.5   2.55     2.1   2.5     2.7    2.7     2    2.55];
minPkVal    = [1e-4 0.35e-3 .75e-3 .75e-3  .4e-3 0.3e-3 0.65e-3 0.55e-3 0.35e-3 0.25e-3 ];

% booleans
removePeaks = 1;   % flag to remove some misidentified peaks
saveData = 1;      % flag to save data

% preallocation
FRFs = [];
fAmps = {};
f0s = {};

% plate 1: delete spurious f0 1 time in the pit
% plate 2: delete initial spurious and 6 peak (has too low amplitude, better off without)
% plate 3: double peak
% plate 4: all good
% plate 5: delete 4th peak, it's spurious
% plate 6: delete 4th peak, and error in the pit
% plate 7: all good
% plate 8: avoid 4th and low one after
% plate 9: avoid double peak at the start
% plate 10: avoid triple peak, pit and double
for plateN = 1:length(H1_matrix(1,:))
    % find peaks
    disp([newline, 'PLATE ' num2str(plateN)])
    [fVals, fLocs] = findpeaks(abs(H1_matrix(:,plateN)),...
                     'minPeakProminence',minPkVal(plateN),...
                     'minPeakWidth', minPkWidths(plateN));
    % save peaks             
    f0 = fAxis(fLocs);
    fAmp = abs(H1_matrix(fLocs, plateN));
    fAmp = fAmp(f0>fMin & f0<fMax);
    f0 = f0(f0>fMin & f0<fMax);
    
    % plot figure
    figure(plateN); clf reset;
    set(gcf, 'Position',  [0, 100, 1500, 400]);
    subplot 212
    p = plot(fAxis, coh_matrix(:,plateN), 'LineWidth', 1.4);
    ylim([0,1.05]); hold on; xline(f0);
    subplot 211
    plot(f0, db(fAmp), 'o', 'markerSize', 5, 'LineWidth', 1.4); hold on ;
    p = plot(fAxis, db(abs(H1_matrix(:,plateN))));
    title(measuredSamples{plateN});

    % remove misidentified peaks
    if removePeaks
        removing = input('how much frequency regions to remove: ');
        for ii = 1:removing
            disp('select low bound of frequency region to delete')
            [fL,y] = ginput(1);
            disp('select high bound of frequency region to delete')
            [fH,y] = ginput(1);
            idxs = find(f0 >= fL & f0 <= fH);
            f0(idxs) = []; fAmp(idxs) = [];
            figure(plateN); clf reset;
            set(gcf, 'Position',  [0, 100, 1500, 400]);
            subplot 212
            p = plot(fAxis, coh_matrix(:,plateN), 'LineWidth', 1.4);
            hold on; xline(f0);
            ylim([0,1.05]);            
            subplot 211
            plot(f0, db(fAmp), 'o', 'markerSize', 5, 'LineWidth', 1.4); hold on ;
            p = plot(fAxis, db(abs(H1_matrix(:,plateN))));
            
        end
    end
     % save peaks into matrices
    f0s{plateN} = f0(:).';
    fAmps{plateN} = fAmp(:).';
    FRFs = [FRFs; H1_matrix(:,plateN).'];
end

% create a struct with 
measFRFs = struct('FRFs', [], 'fAx', [], 'f0s', [], 'fAmps', [], 'pts', []);
measFRFs.FRFs  = FRFs(:,fAxis>= fMin & fAxis <=fMax);  % FRFs signals (n_freq_bins x nPlates) array
measFRFs.fAx   = fAxis(fAxis>= fMin & fAxis <=fMax);   % fAxis        (n_freq_bins x nPlates) array
measFRFs.f0s   = f0s;     % cell array 
measFRFs.fAmps = fAmps;   % cell array
measFRFs.samples   =  measuredSamples;

if saveData 
    cd(baseFolder);
    save('measFRFs', 'measFRFs');
end