function plotFNN(layersLength, layerPosX)

% layersLength is a 1*n row vector containing the number of neurons in each
% layer. E.g., [10,20,5] for a 3 layer NN with 10 inputs and 5 output labels


lengthratio=10*sigmoid(15*rescale(layersLength)-0.4);

totalHeight=1000*sigmoid(15*rescale(layersLength)-0.4);

numThreshold =25;% maximum number of neurons to display in one layer
numShow=layersLength; % to display on top and bottom

L=length(layersLength);
Neurons=NaN*zeros(round(max(totalHeight)),L);
dots=Neurons;

for ii=1:1:L
    
    n=layersLength(ii);
    spots=round(linspace(1,totalHeight(ii),n));
    if n>numThreshold
        numShow(ii)=round(n/6);
        numShow(ii)=min([numShow(ii),8]);
        spots=round(linspace(1,totalHeight(ii),5*numShow(ii)));
        spots(numShow(ii)+1:1:2*numShow(ii))=[];
        mid_loc=mean([spots(numShow(ii)),spots(2*numShow(ii))]);
        dot_loc=round(linspace(mid_loc-60,mid_loc+60,6));
        
        for k=1:1:length(dot_loc)
            dots(dot_loc(k),ii)=dot_loc(k)-mean(dot_loc);
        end
    end
    
    for j=1:1:length(spots)
        Neurons(spots(j),ii)= (spots(j)-mean(spots));
    end
end

syms l [L 1] % neuron height

clearvars l
ls=syms;
NH=[];

for ii = 1:length(ls)
    Neuronstemp=Neurons(:,ii);
    NH.(ls{ii}) = Neuronstemp(~isnan(Neuronstemp));
end

xcor=[];
ycor=[];


for ii=1:1:L-1   
    Li=NH.(ls{ii});
    LiNext=NH.(ls{ii+1});
    
    for j=1:1:length(Li)       
        for k=1:1:length(LiNext)
            x=[layerPosX(ii),layerPosX(ii+1)];
            y=[Li(j),LiNext(k)];           
            xcor=[xcor;x];
            ycor=[ycor;y];            
        end
    end
end


NNplot=figure; axis off, hold on
box on;
xLength = 750;
set(gcf, 'Position',  [0, 50, xLength, 12/20*xLength]);

for ii=1:1:size(xcor,1)
    if ~ismember(xcor(ii,1), 2)
    figure(NNplot)
    plot(xcor(ii,:),ycor(ii,:),'linewidth',0.1,'color',5*[0.1 0.1 0.1])
    hold on
    else
        plot(xcor(ii,:),ycor(ii,:),'-','linewidth',0.1,'color',8*[0.1 0.1 0.1])
    end
    
end

c = colororder;
count = 1;

for ii=1:1:L
    figure(NNplot)
    if ii > 1 && ii<L
    scatter(layerPosX(ii)*ones(size(Neurons(:,ii))),Neurons(:,ii),200,'filled', 'MarkerEdgeColor',[0.2 .2 .2],...
              'MarkerFaceColor',[0.4 .4 .4],...
              'LineWidth',1.5);
    end
    if ii == 1 || ii ==L  
    scatter(layerPosX(ii)*ones(size(Neurons(:,ii))),Neurons(:,ii),600/layersLength(ii),'filled', 'MarkerEdgeColor',0.7*c(count,:),...
            'MarkerFaceColor',c(count,:),'LineWidth',2);
        count = count+1;
    end     
    hold on;
    scatter(layerPosX(ii)*ones(size(dots(:,ii))),dots(:,ii),200, '.','k', 'filled'),hold on
    axis off
    
end
annY =   max(Neurons,[], 'all')+100 ;
fSize = 36;
ylim([min(Neurons,[], 'all') annY])
text(layerPosX(1)-0.05, annY, '$x$', 'interpreter', 'latex', 'fontSize' , fSize)
text(layerPosX(2)-0.05, annY, '$H_1$', 'interpreter', 'latex', 'fontSize' , fSize)
text(layerPosX(3)-0.05, annY, '$H_{\mathcal{L}}$', 'interpreter', 'latex', 'fontSize' , fSize)
text(layerPosX(4)-0.05, annY, '$\hat{y}$', 'interpreter', 'latex', 'fontSize' , fSize)
end

function g = sigmoid(z)
g = 1.0 ./ (1.0 + exp(-z));
end