 function [] = saveRef(ref, refModesNames, modesAnalysisPath, isSum)
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