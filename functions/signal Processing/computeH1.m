function [H1, coh, fAxis] = computeH1(exc, resp, nfft, window, Fs, lowHighcut,algorithm)
    

    % exc    = excitation           --> force vector or matrix    (double)
    % resp   = response             --> response vector or matrix (double)
    % nfft   = number of fft points --> (int)
    % window = tapering window for fft --> use hamm(floor(nfft/n)), choose n
    %          can also define other windows outside before calling the function
    % Fs     = sampling frequency (int)
    % lowHighCut = low and high cut in frequency of the H1 estimator -- 2x1 array [lowCut, higCut]
    % algorithm  = to compute for single vectors set to 'single', otherwise
    %              on matrix 'multiple'
    
    
    if nargin < 7
        algorithm = 'multiple';
    end
    fLow = lowHighcut(1); fHigh = lowHighcut(2);
    
    % compute spectral densities
    [Sxx, f] = pwelch(exc,  window,[],nfft,Fs);
    [Syy, f] = pwelch(resp, window,[],nfft,Fs);
    [Sxy, f] = cpsd(resp, exc, window, [], nfft, Fs);
    [Syx, f] = cpsd(exc, resp, window, [], nfft, Fs);
    
    % work the frequency axis
    cutIdxs = find(f <fHigh & f>fLow );
    fAxis = f(cutIdxs);
    Sxx = Sxx(cutIdxs,:); Syy = Syy(cutIdxs,:);
    Sxy = Sxy(cutIdxs,:); Syx = Syx(cutIdxs,:);
    
    if strcmp(algorithm, 'single')
        H1 =  1./(1i*2*pi*fAxis).*Sxy./Sxx;
        coh = ((abs(Sxy).^2)./(Syy.*Sxx)).';
    end
    
    if strcmp(algorithm, 'multiple')
        H1 =  1./(1i*2*pi*fAxis).*mean(Sxy,2)./mean(Sxx,2);
        coh = ((mean(abs(Sxy),2).^2)./(mean(Syy,2).*mean(Sxx,2))).'; % coherence
    end
end