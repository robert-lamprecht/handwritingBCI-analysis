%This script creates data array for the Mutual Information Analysis that
%investigates the contribution of each electrode 
%This code uses parts of the tSNE code taken from fwillett/handwritingBCI.
%This code was written based on ChatGPT's responses to the following prompts:
%(1)for a matlab array with 27 x 201 x 192 dimensions, how would I write code 
% to average across indices 60:201 in the second dimension to produce a 27 x 192 
% array where each value in the array is the new averaged value?
%(2)I now have 31 arrays of 27 x 192. how to I concatenate them to make a 
% new array of 837x193?

dat = load('singleLetters.mat');

letters = {'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',...
    'greaterThan','comma','apostrophe','tilde','questionMark'};

%%
%Optional: Blockwise Z-scoring
%normalize the neural activity by blockwise z-scoring.  
for x=1:length(letters)
    normCube = single(dat.(['neuralActivityCube_' letters{x}]));
    
    %for each of the 9 blocks, subtract the block-specific mean and then
    %divide by the standard deviation.
    tIdx = 1:3;
    for y=1:9
        mn = zeros(3,1,192);
        mn(1,1,:) = dat.meansPerBlock(y,:);
        mn(2,1,:) = dat.meansPerBlock(y,:);
        mn(3,1,:) = dat.meansPerBlock(y,:);
        
        sd = zeros(1,1,192);
        sd(1,1,:) = dat.stdAcrossAllData;
        
        normCube(tIdx,:,:) = normCube(tIdx,:,:) - mn;
        normCube(tIdx,:,:) = normCube(tIdx,:,:) ./ sd;
        tIdx = tIdx + 3;
    end
    
    dat.(['neuralActivityCube_' letters{x}]) = normCube;
end

%%
% Compute averages for MI

trialavgsforMI = [];  % to hold the concatenated 837x192 data
letteridsforMI = [];   % to hold the identifier column

for f=1:length(letters)
    letterCube = dat.(['neuralActivityCube_' letters{f}]);
 
    averagedlettercube = mean(letterCube(:, 60:201, :), 2);
    averagedletterarray = squeeze(averagedlettercube);
    
    trialavgsforMI_normalized= [trialavgsforMI; averagedletterarray];              % Append 27x192 array
    letteridsforMI = [letteridsforMI; repmat({letters{f}}, 27, 1)];      % Append 27x1 ID column
end


%%
%explort files

save('trialavgsforMI_normalized.mat', 'trialavgsforMI_normalized');
save('letteridsforMI.mat', 'letteridsforMI');


%%
letterCube = dat.(['neuralActivityCube_tilde']);
 
    averagedlettercube = mean(letterCube(:, 60:201, :), 2);
    averagedletterarray = squeeze(averagedlettercube);