function [modesNames, modesNCC] = modeshapes_labeling(pX,pY,...
                        pxFFT, pyFFT, nModes, nTuples, plotFigures, ...
                        ref, refModesNames, compareType, printData)
% ----------------------  modeshapes_labeling -----------------------------
% this function labels the modeshapes of the dataset by comparing them with
% the reference set. For each dataset tuple we compute the Normalised Cross
% Correlation (NCC) and label the given mode of the dataset with the label
% of the reference set comdeshape scoring the best NCC value.
%               -------------------------------------------
% - n.b. this function works both with space domain modeshapes and Fourier
% space domain modeshapes
% -------------------------------------------------------------------------
% inputs: 
% pX  = int - number of points on the x axis of the rectangular grid of modeshapes
% pY  = int - number of points on the y axis of the rectangular grid of modeshapes
% pxFFT = int - number of points on the x axis of the rectangular grid of modeshapes in Fourier domain
% pyFFT = int - number of points on the y axis of the rectangular grid of modeshapes in Fourier domain
% nModes =  int - number of modeshapes to be labeled
% nTuples = int - number of labeled dataset tuples
% plotFigures = boolean - select whether to plot figures while labeling (slow)
% ref           = double array - reference set of modeshapes
% refModesNames = cell array   - labels of the reference set of modeshapes
% compareType   = string - can be "disp" or "fourier", 
%                 selects the type of modeshapes to compare (fourier or displacement)
% printData     = boolean - decide to print some messages while labeling (little bit slower)
% -------------------------------------------------------------------------
% outputs:
% modesNames = cell array - labels of all the modeshapes of the dataset
% modesNCC   = NCC scores associated to each label
% -------------------------------------------------------------------------
 
% colormap
c = [flip(jet)];
 
 % 1) get reference set
 for ii = 1:(length(ref(1,:)))
    modeSurf = reshape(ref(:,ii), [pY, pX]);
    if strcmp(compareType,'fourier')
    refShapes{ii} = fftshift(abs(fft2(modeSurf, pxFFT,pyFFT)));
     elseif strcmp(compareType, 'disp')
    refShapes{ii} = modeSurf;
    else
        disp('input "fourier" or "disp" ');
        return;
    end
 end

 % 2) preallocate variables for labeling
modesAxis = 1:nModes;
dataTuple = 1:nTuples;
tStart = tic;
nPrint = 100;

modesNames_raw = {};

modesNames = cell(nTuples, nModes);
modesNCC = zeros(nTuples, nModes);
modesNMSE = zeros(nTuples, nModes);
maps = zeros(nTuples, nModes);

% 3) label the modeshapes with the reference modeshape scoring maximum ncc
% - for each data tuple
for ii = dataTuple
    if printData, disp([newline, '-------------',  'TUPLE ', num2str(ii), '-------------',newline]); end
    % A) - take modeshapes file of the ii dataset tuple
    modesFilename = ['modeshapes',int2str(ii),'.csv'];
    modesData = readmatrix(modesFilename);
    x = modesData(:,1);     y = modesData(:,2);
    modesData = modesData(:,3:end); % contains the z of all modeshapes
    
    % B) - preallocate the array to store NCC values 
    NCCVec = zeros(length(modesAxis), length(refShapes));
    % C) - for each modeshape associated to the dataset tuple
    % compute NCC between the mode and all reference set modes
    % get the best scoring one
    for jj = modesAxis
       mode_matrix = reshape(modesData(:,jj),[pY,pX]); % get modeshape in matricial form
       mode_vector = abs(mode_matrix(:));              % make it vector form
       for kk = 1:length(refShapes) 
            reference_matrix = refShapes{kk};        % matrix form of generic reference set mode
            reference_vector = reference_matrix(:);  % array form of generic reference set mode
            NCCVec(jj,kk) = abs(NCC(normMinMax(mode_vector), normMinMax(reference_vector)));
       end
       [maxVal, maxLoc] = max(NCCVec(jj,:));
       
       % print labling results
        if printData
            disp(['f_', int2str(jj),' = ', refModesNames{maxLoc},...,
              ' NCC = ', num2str(NCCVec(jj,maxLoc))]); 
        end
       % store best scoring reference mode label 
        modesNames{ii,jj} = refModesNames{maxLoc};
        maps(ii,jj) = maxLoc;
        modesNCC(ii,jj) = NCCVec(jj,maxLoc);
    end
    
        % PLOT figure 
    if plotFigures
        figure(33); 
        clf reset;
        sgtitle(['modes of the ', num2str(ii),'-th tuple']);
        for jj = modesAxis
            mode_matrix = reshape(modesData(:,jj),[pY,pX]);
                figure(33)
                subplot(4,5,jj); imagesc(abs(mode_matrix));     colormap(c);
                title(modesNames{ii,jj}); 
            if strcmp(compareType,'fourier')
                mode_matrix = fftshift(abs(fft2(mode_matrix, pxFFT, pyFFT)));
                figure(34)
                subplot(4,5,jj); imagesc(mode_matrix);
            end
        end
    end
    if mod(ii,nPrint) == 0
        disp(['elapsed time for ', int2str(ii), ' tuples: ', num2str(round(toc(tStart),1)), ' seconds'])  ;
        end
end
end

function [normF] = normMinMax(fun)
normF = (fun - min(fun))./(max(fun) - min(fun));
end