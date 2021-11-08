function [] = resampleMshapes(pX,pY,nModes, nTuples,...
    resampledFolder, plotData, modesFilename)
if nargin <7
    modesFilename = 'modeshapes';
end
mkdir(resampledFolder);
cd(resampledFolder)
m = readmatrix([modesFilename '1.csv']);

rangeX = max(m(:,1))-min(m(:,1)); rangeY = max(m(:,2))-min(m(:,2));
xMin = min(m(:,1))+0.01*rangeX;    xMax = 0.97*max(m(:,1));
yMin = min(m(:,2))+0.01*rangeY;    yMax = 0.92*max(m(:,2));
[X,Y] = meshgrid(linspace(xMin, xMax, pX), linspace(yMin, yMax, pY));

x = X(:); y = Y(:);
c = [jet; flip(jet)];
for ii = 1:nTuples
    modeshapes = [x,y];
    m = readmatrix(['modeshapes', int2str(ii),'.csv']);
    for jj = 1:nModes
        F1 = scatteredInterpolant(m(~isnan(m(:,1)),1),m(~isnan(m(:,1)),2), m(~isnan(m(:,1)),3+jj), 'linear');
        Z = F1(X,Y);
        if plotData
            figure(1)
            surf(X,Y,Z); %view(2);
            colormap(c);
            colorbar;
            view(3);
            pause(0.01)
        end
        z = Z(:);
        modeshapes = [modeshapes, z];
    end
    modeshapes = sortrows(modeshapes);
    writeMat2File(modeshapes, ['modeshapes', int2str(ii),'.csv'], {'f'}, 1,false);
end
end