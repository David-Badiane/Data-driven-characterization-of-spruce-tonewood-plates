function [modesNames, modesNCC, modesNMSE] = modeshapesLabeling(pX,pY,...
                        pxFFT, pyFFT, nModes, nTuples, plotFigures, ...
                        ref, refModesNames, compareType, printData)
c = [jet; flip(jet)];
                            
 for ii = 1:(length(ref(1,:)))
    modeSurf = reshape(ref(:,ii), [pY, pX]);
    refShapes_FFT{ii} = fftshift(abs(fft2(modeSurf, pxFFT,pyFFT)));
    refShapes_Disp{ii} = modeSurf;
 end

 if strcmp(compareType,'fourier')
     refShapes = refShapes_FFT;
 elseif strcmp(compareType, 'disp')
     refShapes = refShapes_Disp;
 else
    disp(' "fourier" or "disp" ');
    return;
 end
                             
modesAxis = 1:nModes;
modesNames = cell(nTuples, nModes);
modesNCC = zeros(nTuples, nModes);
modesNMSE = zeros(nTuples, nModes);
dataTuple = 1:nTuples;
tStart = tic;
nPrint = 100;
for ii = dataTuple
    
    modesFilename = ['modeshapes',int2str(ii),'.csv'];
    modesData = readmatrix(modesFilename);
    x = modesData(:,1);     y = modesData(:,2);
    modesData = modesData(:,3:end);
    
    if mod(ii,nPrint) == 0
    disp(' ');
    end
    if plotFigures
    figure(33)
    sgtitle(['modes of the ', num2str(ii),'-th tuple']);
    end
%     disp(ii);
    for jj = modesAxis
         
            modeMatrix = reshape(modesData(:,jj),[pY,pX]);
            if plotFigures
                figure(33)
                subplot(4,5,jj); imagesc(modeMatrix);     colormap(c);
            end
            if strcmp(compareType,'fourier')
            modeMatrix = fftshift(abs(fft2(modeMatrix, pxFFT, pyFFT)));
            if plotFigures
                figure(34)
                subplot(4,5,jj); imagesc(modeMatrix);
            end
            end
        
        lossFx = zeros(size(refShapes));
        NCCVec = zeros(size(refShapes));
        NMSEVec = zeros(size(refShapes));
        modeVec = modeMatrix(:);
        
        for kk = 1:length(refShapes) 
            testMatrix = refShapes{kk};
            testVec = testMatrix(:);
            NCCVec(kk) = NCC(modeVec, testVec);
            NMSEVec(kk) = NMSE(modeVec, testVec);
            lossFx(kk) = NCCVec(kk) - NMSEVec(kk) ;
        end
        [maxVal, maxLoc] = max(lossFx);
       
        if printData
        disp(['f_', int2str(jj),' = ', refModesNames{maxLoc}, ' NCC = ', num2str(NCCVec(maxLoc)), ' NMSE = ',num2str(NMSEVec(maxLoc))]); 
        end
        if ii == 30
            a =1; 
        end
        modesNames{ii,jj} = refModesNames{maxLoc};
        if plotFigures
            figure(33); title(refModesNames{maxLoc});
        end
        modesNCC(ii,jj) = NCCVec(maxLoc);
        modesNMSE(ii,jj) = NMSEVec(maxLoc);
    end
    if mod(ii,nPrint) == 0
    disp(['elapsed time for ', int2str(ii), ' tuples: ', num2str(round(toc(tStart),1)), ' seconds'])  ;
    end
end


end