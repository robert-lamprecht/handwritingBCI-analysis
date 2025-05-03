%This script was taken from fwillett/handwritingBCI. We made
%minor changes to the author's original code and included additional
%annotations. 
%We used chatGPT to edit this code with the following prompts:
%(1) Edit this code so that it plots single data points instead of the letters 
% but labels the cluster with the corresponding letter somewhere on the graph. 
% Plot the letters right next to each cluster instead of on top. 
%(2) Average the PCA scores across the 142 time points so that for each row 
% there are 15 columns (one for each PC).


dat = load('singleLetters.mat');

letters = {'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',...
    'greaterThan','comma','apostrophe','tilde','questionMark'};

%%
%normalize the neural activity by blockwise z-scoring. 
for x=1:length(letters)
    normCube = single(dat.(['neuralActivityCube_' letters{x}])); %this loads 
    % the cube of 27 trials x 201 time bins x 192 electrodes
    
    %for each of the 9 blocks, subtract the block-specific mean and then
    %divide by the standard deviation.
    tIdx = 1:3; %trial index
    for y=1:9
        mn = zeros(3,1,192);
        mn(1,1,:) = dat.meansPerBlock(y,:); %meansperblock is a 9 x 192 
        % structure (means of 3 rows/trials)
        mn(2,1,:) = dat.meansPerBlock(y,:);
        mn(3,1,:) = dat.meansPerBlock(y,:);
        
        sd = zeros(1,1,192);
        sd(1,1,:) = dat.stdAcrossAllData; %std of activity for every 
        % electrode (192 total)
        
        %calculate the z-score ((x-mn)/std)
        normCube(tIdx,:,:) = normCube(tIdx,:,:) - mn;
        normCube(tIdx,:,:) = normCube(tIdx,:,:) ./ sd;
        tIdx = tIdx + 3; %This function allows code to loop through all 
        % 9 groups of 3 trials
    end
    
    dat.(['neuralActivityCube_' letters{x}]) = normCube;
end

%%
%compute trial-averaged activity for each character, using a 50ms sd
%gaussian smoothing kernel. 
allData = zeros(2000,27264);
allSpatial = zeros(200000,192); 
allLabels = zeros(2000,1);
allAvg = [];
cIdx = 1;
spatialIdx = 1;

for f=1:length(letters)
    letterCube = dat.(['neuralActivityCube_' letters{f}]);
    for x=1:size(letterCube,1)
        row = gaussSmooth_fast(squeeze(letterCube(x,60:end,:)), 5);

        row = row(:);
        allData(cIdx,:) = row; 
        allLabels(cIdx,:) = f;
        cIdx = cIdx + 1; %move to next index in loop

        %timeseries (time bins in letter cube and width (width = 50 ms))
        newChunk = gaussSmooth_fast(squeeze(letterCube(x,60:end,:)), 5); 
        allSpatial(spatialIdx:(spatialIdx+size(newChunk,1)-1),:) = newChunk;
        spatialIdx = spatialIdx + size(newChunk,1);
    end
    
    avgLet = squeeze(mean(letterCube,1));
    avgLet = gaussSmooth_fast(avgLet(60:end,:), 5);
    allAvg = [allAvg; avgLet];
end

%%
%use the trial-averaged activity to compute PCs, then take the top 15 PCs
%as features for the single-trial data, using a 50 ms sd smoothing kernel.
[COEFF, SCORE, LATENT, TSQUARED, EXPLAINED, MU] = pca(allAvg);

nDim = 15; %number of PC dimensions (Willett et al. (2021) used 15)
allData = zeros(2000,142*nDim);
allLabels = zeros(2000,1);
cIdx = 1; %character index (31 total)
spatialIdx = 1;

for f=1:length(letters)
    tmp = dat.(['neuralActivityCube_' letters{f}]);
    for x=1:size(tmp,1) %loops through each row
        row = gaussSmooth_fast(squeeze(tmp(x,60:end,:)), 5); %use 5 for 50ms smoothing kernel
        row = (row-MU)*COEFF(:,1:nDim);%transform for specific PC (COEFF = eigenvectors, MU = mean)
         
        row = row(:);
        allData(cIdx,:) = row;
        allLabels(cIdx,:) = f;
        cIdx = cIdx + 1;
    end
end

allData(isnan(allData)) = 0;
allData = allData(1:(cIdx-1),:);
allLabels = allLabels(1:(cIdx-1));

%%
%tsne using warp-distance, this could take ~10-15 minutes. 
nTimeBinsPerPoint = 142;
warpDistFun = @(d1,d2)(tsneWarpDist(d1,d2,142));
[Y,loss] = tsne(allData,'Distance',warpDistFun,'Verbose',2,'Perplexity',40); 
%Y is the output of tSNE used for visualization. 40 means that t-SNE will 
% consider 40 neighbors when building probability distribution

%%
%plot tsne results
plotLet = {'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',...
    '>',',','''','~','?'};

colors = [    0.3613    0.8000         0
    0.8000         0    0.1548
    0.8000    0.1548         0
    0.8000         0    0.4645
    0.6710         0    0.8000
    0.3613         0    0.8000
    0.5161    0.8000         0
    0.8000    0.6194         0
    0.8000         0         0
    0.6710    0.8000         0
    0.2065         0    0.8000
         0    0.1032    0.8000
    0.8000    0.3097         0
         0    0.7226    0.8000
         0    0.8000    0.5677
         0    0.8000    0.2581
         0    0.2581    0.8000
    0.0516         0    0.8000
    0.8000         0    0.6194
    0.8000    0.4645         0
    0.8000         0    0.7742
    0.8000    0.7742         0
         0    0.5677    0.8000
    0.8000         0    0.3097
         0    0.8000    0.4129
    0.0516    0.8000         0
         0    0.4129    0.8000
    0.5161         0    0.8000
    0.2065    0.8000         0
         0    0.8000    0.1032
         0    0.8000    0.7226];

ylims = [min(Y(:,2)), max(Y(:,2))];
xlims = [min(Y(:,1)), max(Y(:,2))];

figure('Color','w');
hold on;
for x=1:size(Y,1)
    text(Y(x,1), Y(x,2), plotLet{allLabels(x)},'Color',colors(allLabels(x),:),'FontWeight','bold','FontSize',6);
end
xlim(xlims);
ylim(ylims);
axis equal;
axis off;


%%
% plot tsne results with points and cluster labels -- edited version
plotLet = {'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',...
    '>',',','''','~','?'};

colors = [    0.3613    0.8000         0
    0.8000         0    0.1548
    0.8000    0.1548         0
    0.8000         0    0.4645
    0.6710         0    0.8000
    0.3613         0    0.8000
    0.5161    0.8000         0
    0.8000    0.6194         0
    0.8000         0         0
    0.6710    0.8000         0
    0.2065         0    0.8000
         0    0.1032    0.8000
    0.8000    0.3097         0
         0    0.7226    0.8000
         0    0.8000    0.5677
         0    0.8000    0.2581
         0    0.2581    0.8000
    0.0516         0    0.8000
    0.8000         0    0.6194
    0.8000    0.4645         0
    0.8000         0    0.7742
    0.8000    0.7742         0
         0    0.5677    0.8000
    0.8000         0    0.3097
         0    0.8000    0.4129
    0.0516    0.8000         0
         0    0.4129    0.8000
    0.5161         0    0.8000
    0.2065    0.8000         0
         0    0.8000    0.1032
         0    0.8000    0.7226];

ylims = [min(Y(:,2)), max(Y(:,2))];
xlims = [min(Y(:,1)), max(Y(:,1))];

figure('Color','w');
hold on;

% plot points
for x = 1:size(Y,1)
    clusterIdx = allLabels(x);
    plot(Y(x,1), Y(x,2), '.', 'Color', colors(clusterIdx,:), 'MarkerSize', 8);
end

% compute and label cluster centroids with offset
offset = [0.5, 0.3]; % adjust as needed
uniqueLabels = unique(allLabels);
for i = 1:length(uniqueLabels)
    idx = uniqueLabels(i);
    clusterPoints = Y(allLabels == idx, :);
    centroid = mean(clusterPoints, 1);
    text(centroid(1) + offset(1), centroid(2) + offset(2), plotLet{idx}, ...
        'Color', 'k', ...
        'FontWeight', 'bold', ...
        'FontSize', 20, ...
        'HorizontalAlignment', 'left', ...
        'VerticalAlignment', 'bottom');
end

xlim(xlims);
ylim(ylims);
axis equal;
axis off;

%%
%k nearest neighbor classificaiton using warp-distance, this could take a few
%minutes to compute D. k=10
D = pdist(allData, warpDistFun); %computes distances between allData (from PCA) and the time warped version of allData
D = squareform(D);

classAcc = zeros(size(D,1),1);
for x=1:length(classAcc)
   
    [~,sortIdx] = sort(D(x,:)); 
    sortIdx = sortIdx(2:11); %selects 10 nearest neighbors
    choice = mode(allLabels(sortIdx)); %finds identity of nearest neighbors 

    classAcc(x) = choice==allLabels(x); %checks if the choice is the same as the label
end

disp('Warp NN Accuracy (CI)');
disp(mean(classAcc)); %classification accuracy
[PHAT, PCI] = binofit(sum(classAcc),length(classAcc)); %binomial confidence interval
disp(PCI);



%%
%transform the output of the PCA to 15 x 837 matrix for Mutual information
%analysis by averaging activity across time bins
allData_forMI = squeeze(mean(reshape(allData, [], 142, nDim), 2));