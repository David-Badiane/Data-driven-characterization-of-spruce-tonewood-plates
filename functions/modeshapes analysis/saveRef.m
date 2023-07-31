 function [] = saveRef(ref, refModesNames, modesAnalysisPath, isSum)
 % saves the reference set of modes
 % ---------------------
 % INPUTS:
 % ref               = (nPts x nRefModes) 2DArray
 %                     reference set of modeshapes
 % refModesNames     = cell array 
 %                     labels (e.g. f_02) associated to the reference set
 % modesAnalysisPath = string
 %                     path to the modes analysis directory of csv_gPlates
 % isSum <--- deprecated, ignore it
 % ---------------------
 % OUTPUTS:
 % ~
if nargin<4, isSum = 0; end
    
 cd(modesAnalysisPath)
    reVarNames = {};
    addStr = {'_a' '_b' '_c' '_d' '_e' '_f' };
    for ii = 1:length(refModesNames)
    repetitions = find(ismember({refModesNames{:}}, refModesNames(ii)));
    nRepetition = find(repetitions == ii);
    reVarNames{ii} = [refModesNames{ii}, addStr{nRepetition}];
    end
    if isSum
        writeMat2File(ref, ['reference_sum.csv'], reVarNames, length(reVarNames), true);
        writeMat2File(refModesNames, 'refModesNames_sum.csv', {'name '}, 1,false);
    else
        writeMat2File(ref, ['reference.csv'], reVarNames, length(reVarNames), true);
        writeMat2File(refModesNames, 'refModesNames.csv', {'name '}, 1,false);
    end
end