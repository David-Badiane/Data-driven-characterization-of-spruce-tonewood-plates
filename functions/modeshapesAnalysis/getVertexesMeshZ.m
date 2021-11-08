function [outPts] = getVertexesMeshZ(pts, stepZ,scaleMesh)

   x = pts(:,1)*scaleMesh; y = pts(:,2)*scaleMesh; z = pts(:,3)*scaleMesh;
   maxX = max(x)
   minX = min(x);
   maxY = max(y)
   minY = min(y);
   maxZ = max(z) 
   minZ = min(z);
   
   
   zVals = minZ:stepZ:maxZ;
   nPts = 20;
   xAx = linspace(minX, maxX, nPts);
   yAx = linspace(minY, maxY, nPts);
   zAx = linspace(minZ, maxZ, nPts);
   
   figure(4); hold on;
   edges = [];
   
   plotData = 0; 
   syms xx yy;
   outPts = [];
%    maxSearch = logspace(log10(0.001), log10(0.08), length(zVals)-1);
   maxSearch = 0.0075* ones(length(zVals)-1,1);
   for ii = 1:length(zVals)-1
        idxsZ = find(z > zVals(ii) & z < zVals(ii+1));
        xZ = x(idxsZ); yZ = y(idxsZ); zZ = z(idxsZ);
        
        mM = [min(xZ) max(xZ) min(yZ) max(yZ) min(zZ) max(zZ)];
        ptsZ = [xZ, yZ, zZ];

%         plot3(xZ,yZ,zZ, '.');
        % xy plane
        
        % edge 1
        p1 = fitEdge(ptsZ, 'y', 'min', mM(3:4), plotData,pts, maxSearch(ii));
        % edge 2
        p2 = fitEdge(ptsZ, 'x', 'max', mM(1:2), plotData,pts, maxSearch(ii));
        % edge 3
        p3 = fitEdge(ptsZ, 'y', 'max', mM(3:4), plotData,pts, maxSearch(ii));
        % edge 4
        p4 = fitEdge(ptsZ, 'x', 'min', mM(1:2), plotData,pts, maxSearch(ii));
        
        edges = {p1 p2 p3 p4};

        for jj =1:length(edges)  
            A = [edges{jj}.XY(1)                    -1;...
                 edges{mod(jj,length(edges))+1}.XY(1)  -1];
            B = [-edges{jj}.XY(2);...
                 -edges{mod(jj,length(edges))+1}.XY(2)];
            sol = linsolve(A,B);
            outPts = [outPts; sol.' mean(zZ)];           
        end
        
   if ii >= floor(0.2*length(zVals))
       a = 1;
   end   
   end

   
    figure(4);
    subplot(121); hold on; box on;
    plot3(outPts(:,1), outPts(:,2), outPts(:,3), 'lineWidth', 1.1)
    plot3(outPts(:,1), outPts(:,2), outPts(:,3), '.', 'MarkerSize', 10)
    view(3);
    xlabel('x [mm]'); ylabel('y [mm]'); zlabel('z [mm]');
    ylim([min(outPts(:,2)),130])
    ax = gca;
    ax.FontSize = 12;
    ax.LineWidth = 0.65;
    xlim([-10,500]);
    ylim([-10,200]);
    zlim([-5,45]);
end

function [p] = fitEdge(pts, variable, edge, minMax, plotData, allPts, maxSearch)
    p = struct('xAx', [], 'XY', []);
    
    idxFit = find(strcmp(variable, {'x' 'y' 'z'}));
    idxGet = [2,1,3];
    idxGet = idxGet(idxFit);
    nPoints = 30;
    mult = .015;
    
   if strcmp(edge, 'min')
   idx = find(pts(:,idxGet)<(min(pts(:,idxGet)) + mult*abs(min(pts(:,idxGet)))));
   maxV = max(pts(idx,idxFit));
   minV = min(pts(idx,idxFit));
   elseif strcmp(edge, 'max')
       idx = find(pts(:,idxGet)>(1-mult)*max(pts(:,idxGet)));
   maxV = max(pts(idx,idxFit));
   minV = min(pts(idx,idxFit));
   end

    axsearch = linspace(minV, maxV, nPoints);
    edgeIdxs = [];
    deltaAx = axsearch(2)-axsearch(1);
    for ii = 1:length(axsearch)-1
       idxs = find(pts(:,idxFit)<axsearch(ii+1) & pts(:,idxFit)>axsearch(ii));
       if strcmp(edge, 'min')
           [mV,mL] = min(pts(idxs,idxGet));
        elseif strcmp(edge, 'max')
           [mV,mL] = max(pts(idxs,idxGet));
       end  
       edgeIdxs = [edgeIdxs; idxs(mL)];
%        figure(6); hold on;
%        plot3(pts(idxs,1),pts(idxs,2),pts(idxs,3), '.', 'markerSize', 0.4)
%        plot3(pts(edgeIdxs,1),pts(edgeIdxs,2),pts(edgeIdxs,3), '.', 'markerSize', 15)
%        view(2);
    end
    
    if strcmp(edge, 'min')
        edgeIdxs = find(pts(:,idxFit) < minMax(1)+maxSearch*(minMax(2) - minMax(1)) );
    elseif strcmp(edge, 'max')
        edgeIdxs = find(pts(:,idxFit) > minMax(2)-maxSearch*(minMax(2) - minMax(1)) );    
    end  
    
    p.XY = polyfit(pts(edgeIdxs,1),pts(edgeIdxs,2),1).';
    
    if idxFit == 1
        yStep =  max(allPts(:,2))-min(allPts(:,2));
        yAx = linspace(min(allPts(:,2))-0.3*yStep, max(allPts(:,2))+0.3*yStep,100);
        xAx = polyval(polyfit(pts(edgeIdxs,2),pts(edgeIdxs,1),1).', yAx);
    end
    
    if idxFit == 2
        xStep = max(allPts(:,1))-min(allPts(:,1));
        xAx = linspace(min(allPts(:,1))-0.3*xStep, max(allPts(:,1))+0.3*xStep,100); 
    end
    
    p.xAx = xAx;
    pxy = polyval(p.XY, xAx);
    if plotData
        figure(3); 
        hold on;
        plot3(xAx, pxy, mean(pts(:,3))*ones(size(pxy)));
    end
    
      if plotData 
        figure(1); clf reset;
        hold on;
        plot3(pts(:,1), pts(:,2), pts(:,3), '.', 'markerSize', 1);
        plot3(xAx, pxy, mean(pts(:,3))*ones(size(pxy)));
        plot3(pts(edgeIdxs, 1),pts(edgeIdxs, 2),pts(edgeIdxs, 3),'.') ;
    end
    
end


