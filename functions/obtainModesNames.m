function [modesNames, namesTable] = obtainModesNames(nSim, nModes,minimumThreshold, csvPath,simFolder, modesFilename)
%ANALYSEMODESHAPES Summary of this function goes here
%   Detailed explanation goes here
    
    cd(simFolder)
    modesNames = cell(nSim,nModes);

    for ii = 1:nSim
        meshData = table2array(readtable(['mesh',int2str(ii),'.csv']));
        modesData = table2array(readtable(['modeshapes',int2str(ii),'.csv']));
        [modesNamesSim] = recognizeModes(meshData,modesData,minimumThreshold^-1, ii );
        modesNames(ii,1:nModes) = modesNamesSim;
    end

    cd(csvPath)
    % comment this if you want to obtain modeshape names
    namesTable  = writeMat2File(modesNames,modesFilename, {'f'}, 1,false);
end

