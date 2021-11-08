function vertexesXY = getFinalVertexesXY(edges)
idxIntersection = [1,2; 2,3; 3,4; 4,1; 5,6; 6,7; 7,8; 8,5;];
vertexesXY = [];    
    for ii = 1:length(idxIntersection(:,1))
        A = [edges(1,idxIntersection(ii,1))    -1;...
             edges(1,idxIntersection(ii,2))    -1];
        B = [-edges(2,idxIntersection(ii,1));...
                 -edges(2,idxIntersection(ii,2))];
        sol = linsolve(A,B);
        vertexesXY = [vertexesXY; sol.' ]; 
    end  
end