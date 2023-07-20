function [] = modeshapes_resampling(pX,pY,nCols, nTuples,...
    resampledFolder, plotData)
% RESAMPLEMSHAPES
% this function allows to resample modeshapes from Comsol irregular
% rectangular grid to a regular rectangular grid
% copyright: David Giuseppe Badiane
% -------------------------------------------------------------------------
% inputs:
%   pX               = int - number of points along the x axis of the
%                          rectangular grid
%   pY               = int - number of points along the y axis of the
%                          rectangular grid
%   nCols            = int - number of dataset columns taken into account (how many eigenfrequencies)
%   nTuples          = int - number of dataset tuples
%   resampledFolder  = string - path where to save resampled modeshapes
%   plotData         = boolean - select wheter to plot figures while
%                              resampling (slower)
% -------------------------------------------------------------------------
% outputs:
% ~
% -------------------------------------------------------------------------
    cd(resampledFolder)
    
    % name of modeshapes files
    modesTag = 'modeshapes';
    % read the first
    m = readmatrix([modesTag, '1.csv']);
    % get variables names a.k.a. columns names
    varNames = {'x' 'y'};
    for ii = 1:nCols
        varNames = {varNames{:} ['f', int2str(ii)]};
    end
    
    % variables
    % figure
    c = [flip(jet);]; % color map for the figure
    % subplot numbers
    px = round(sqrt(nCols)); py = px+1;
    delta = 0.01;

    % timer
    t = tic;
    
    for ii = 1:nTuples
        % 0) fetch modeshapes file
        m = readmatrix([modesTag,  int2str(ii),'.csv']);
        % 1) normalize
        m(:,1) = minmaxNorm(m(:,1));
        m(:,2) = minmaxNorm(m(:,2));
        % 1.1) take min and max
        xMin = min(m(:,1));    xMax = max(m(:,1));
        yMin = min(m(:,2));    yMax = max(m(:,2));

        % 2) define a rectangular grid
        [X,Y] = meshgrid(linspace(xMin, xMax, pX), linspace(yMin, yMax, pY));
        x = X(:); y = Y(:);
        modeshapes = [x,y]; % here we will save resampled modeshapes!
        
        % open the figure if set to do so
        if plotData, figure(2); clf reset; disp(['tuple ', int2str(ii)]); end
        
        % 3) Interpolate for each modeshape
        for jj = 1:nCols
            F1 = scatteredInterpolant(m(:,1),m(:,2), m(:,3+jj), 'natural', 'nearest');
            Z = F1(X,Y);
%           Z = griddata(m(:,1),m(:,2),m(:,3+jj),X,Y,'v4'); % better but way slower
            
            % fill the figure
            if plotData
                figure(2); 
                subplot(px, py, jj)
                surf(X,Y,Z);
                colormap(c);
                shading interp;
                view(2);
            end
            % save resampled modeshape in array 
            z = Z(:);
            modeshapes = [modeshapes, z];
        end
        
        % 4) save resampled modeshapes into a file
        % sort the modeshapes by the x value
        modeshapes = sortrows(modeshapes);
        % write file 
        writeMat2File(modeshapes, [modesTag, int2str(ii),'.csv'], varNames, length(varNames),true);
        
        % a counter each 100 resampled dataset tuples
        if mod(ii,100) == 0
            disp(['elapsed time for', num2str(ii), ' tuples:',...
                num2str(floor(toc(t)/60)) 'm ', num2str(round(mod(toc(t),60))), 's'])
        end
   end
end


function normOut = minmaxNorm(normIn)
normOut = (normIn-min(normIn))./(max(normIn)-min(normIn));
end

function surf = smoothSurf(surf, nTimes)
    for kk = 1:nTimes
        for ii = 1:size(surf,1)
            for jj = 1:size(surf,2)
                if ii == 1
                  vect1 = surf(ii+1,jj);
                elseif ii == size(surf,1)
                  vect1 = surf(ii-1,jj);
                else
                  vect1 = [surf(ii+1,jj) surf(ii-1,jj)];
                end
                if jj == 1
                  vect2 = surf(ii,jj+1);
                elseif jj == size(surf,2)
                  vect2 = surf(ii,jj-1);
                else
                vect2 = [surf(ii,jj+1) surf(ii,jj-1)];
                end
                vect = [vect1, vect2];
                surf(ii,jj) = mean(vect);
            end
        end
    end
end