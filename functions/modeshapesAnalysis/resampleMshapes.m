function [] = resampleMshapes(pX,pY,nModes, nTuples,...
    resampledfolder, csvPath, plotData)

resampledFolder = [csvPath,'\', resampledfolder];
mkdir(resampledFolder);
cd(resampledFolder)
m = readmatrix(['modeshapes1.csv']);
xMin = min(m(:,1));    xMax = max(m(:,1));
yMin = min(m(:,2));    yMax = max(m(:,2));
[X,Y] = meshgrid(linspace(xMin, xMax, pX), linspace(yMin, yMax, pY));

x = X(:); y = Y(:);

for ii = 1:nTuples
    modeshapes = [x,y];
    m = readmatrix(['modeshapes', int2str(ii),'.csv']);
    for jj = 1:nModes
        F1 = scatteredInterpolant(m(:,1),m(:,2), m(:,3+jj), 'natural');
        Z = F1(X,Y);
        if plotData
            figure(1)
            surf(X,Y,Z); view(2);
        end
        z = Z(:);
        modeshapes = [modeshapes, z];
    end
    modeshapes = sortrows(modeshapes);
    writeMat2File(modeshapes, ['modeshapes', int2str(ii),'.csv'], {'f'}, 1,false);
end
end