function [H1, coh, fAxis] = computeH1(force, acceleration, nfft, window, Fs, lowHighcut,algorithm)
% COMPUTE H1
% this function computes the h1 estimator of the mobility (velocity/force) 
% from force and acceleration measurements
% -------------------------------------------------------------------------
% inputs:
% exc    = excitation           --> force vector or matrix    (double)
% resp   = response             --> response vector or matrix (double)
% nfft   = number of fft points --> (int)
% window = tapering window for fft --> use hamm(floor(nfft/n)), choose n
%          can also define other windows outside before calling the function
% Fs     = int - sampling frequency in Hz
% lowHighCut = 2x1 array - low and high cut in frequency of the H1 
%                          estimator [lowCut, highCut]
% algorithm  = to compute H1 for single vectors set to 'single', otherwise
%              on matrix 'multiple'
% -------------------------------------------------------------------------
% outputs:
% H1     = array - H1 estimator of the mobility
% coh    = array - coherence associated to the H1 estimator
% fAxis  = array - frequency axis associated to the H1 estimator
% ------------------------------------------------------------------------- 
    if nargin < 7
        algorithm = 'multiple';
    end
      
    % compute spectral densities
    [Sxx, f] = pwelch(force,  window,[],nfft,Fs);
    [Syy, f] = pwelch(acceleration, window,[],nfft,Fs);
    [Sxy, f] = cpsd(acceleration, force, window, [], nfft, Fs);
    [Syx, f] = cpsd(force, acceleration, window, [], nfft, Fs);
    
    % cut the frequency axis to the bandwidth [fLow, fHigh]
     fLow = lowHighcut(1); fHigh = lowHighcut(2);
    cutIdxs = find(f <fHigh & f>fLow );
    fAxis = f(cutIdxs);
    Sxx = Sxx(cutIdxs,:); Syy = Syy(cutIdxs,:);
    Sxy = Sxy(cutIdxs,:); Syx = Syx(cutIdxs,:);
    
    % compute H1 estimator
    if strcmp(algorithm, 'single')
        H1 =  1./(1i*2*pi*fAxis).*Sxy./Sxx;
        coh = ((abs(Sxy).^2)./(Syy.*Sxx)).';
    end
    
    if strcmp(algorithm, 'multiple')
        H1 =  1./(1i*2*pi*fAxis).*mean(Sxy,2)./mean(Sxx,2);
        coh = ((mean(abs(Sxy),2).^2)./(mean(Syy,2).*mean(Sxx,2))).'; % coherence
    end
end