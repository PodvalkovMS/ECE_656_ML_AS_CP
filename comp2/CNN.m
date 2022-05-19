clc
clear

%Download the CIFAR-10 dataset

if ~exist('cifar-10-batches-mat','dir')
    cifar10Dataset = 'cifar-10-matlab';
    disp('Downloading 174MB CIFAR-10 dataset...');   
    websave([cifar10Dataset,'.tar.gz'],...
        ['https://www.cs.toronto.edu/~kriz/',cifar10Dataset,'.tar.gz']);
    gunzip([cifar10Dataset,'.tar.gz'])
    delete([cifar10Dataset,'.tar.gz'])
    untar([cifar10Dataset,'.tar'])
    delete([cifar10Dataset,'.tar'])
end    
   
% Prepare the CIFAR-10 dataset
if ~exist('cifar10Train','dir')
    disp('Saving the Images in folders. This might take some time...');    
    save_cifar_10_as_folder_of_images('cifar-10-batches-mat', pwd, true);
end

% Load image CIFAR-10 Training dataset (50000 32x32 colour images in 10 classes)
imsetTrain = imageSet('cifar10Train','recursive');


% Prepare the data for Training
% Read all images and store them in a 4D uint8 input array for training,
% with its corresponding class

trainNames = {imsetTrain.Description};
XTrain = zeros(32,32,3, sum([imsetTrain.Count]*0.8), 'uint8');
XVal = zeros(32,32,3, sum([imsetTrain.Count]*0.2), 'uint8');
TTrain = categorical(discretize((1:sum([imsetTrain.Count]*0.8))',...
    [0,cumsum([imsetTrain.Count]*0.8)],'categorical',trainNames));
TVal = categorical(discretize((1:sum([imsetTrain.Count]*0.2))',...
    [0,cumsum([imsetTrain.Count]*0.2)],'categorical',trainNames));


ere=0
epe=0
tic;
for c = 1:length(imsetTrain)
    for i = 1:imsetTrain(c).Count
        if i<=imsetTrain(c).Count*0.8
        epe=epe+1    
        XTrain(:,:,:, epe) =read(imsetTrain(c),i);

        else
        ere=ere+1;
        XVal(:,:,:, ere) =read(imsetTrain(c),i);
      
        end
    end
    
end

toc;

% Define a CNN architecture
conv1 = convolution2dLayer(3,32,'Padding',2,...
                     'BiasLearnRateFactor',2);
conv1.Weights = gpuArray(single(randn([3 3 3 32])*0.0001));


layers1 = [
    imageInputLayer([32 32 3])
    
    conv1 
    maxPooling2dLayer(2,'Stride',2)
    reluLayer()
    
    convolution2dLayer(3,64,'Padding','same')
    maxPooling2dLayer(2,'Stride',2)
    reluLayer()
    
    convolution2dLayer(3,128,'Padding','same')
    maxPooling2dLayer(2,'Stride',2)
    reluLayer()
    
    fullyConnectedLayer(128)
    reluLayer()
    fullyConnectedLayer(10)
    softmaxLayer()
    classificationLayer()];


layers2 = [
    imageInputLayer([32 32 3])
    
    conv1 
    maxPooling2dLayer(2,'Stride',2)
    reluLayer()
    
    convolution2dLayer(3,64,'Padding','same')
    maxPooling2dLayer(2,'Stride',2)
    reluLayer()
    
    convolution2dLayer(3,64,'Padding','same')
    maxPooling2dLayer(2,'Stride',2)
    reluLayer()
    
    fullyConnectedLayer(64)
    reluLayer()
    fullyConnectedLayer(10)
    softmaxLayer()
    classificationLayer()];


layers3 = [
    imageInputLayer([32 32 3])
    
    conv1 
    maxPooling2dLayer(2,'Stride',2)
    reluLayer()
    
    convolution2dLayer(3,32,'Padding','same')
    maxPooling2dLayer(2,'Stride',2)
    reluLayer()
    
    convolution2dLayer(3,64,'Padding','same')
    maxPooling2dLayer(2,'Stride',2)
    reluLayer()
    
    fullyConnectedLayer(64)
    reluLayer()
    fullyConnectedLayer(10)
    softmaxLayer()
    classificationLayer()];


% Define the training options.
opts = trainingOptions("sgdm", ...
    'InitialLearnRate', 0.001, ...
    'LearnRateSchedule',"piecewise", ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 8, ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 100, ...
    'ValidationData', {XVal,TVal}, ...
    'ValidationFrequency', 40, ...
    'Plots', 'training-progress');

% Define the training options.
opts2 = trainingOptions('rmsprop', ...
    'InitialLearnRate', 0.001, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 8, ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 100, ...
    'ValidationData', {XVal,TVal}, ...
    'ValidationFrequency', 40, ...
    'Plots', "training-progress");

% Training the CNN
layers=[layers1, layers2, layers3]
[net, info] = trainNetwork(XTrain, TTrain, layers(:,1), opts);
[net2, info2] = trainNetwork(XTrain, TTrain, layers(:,2), opts);
[net3, info3] = trainNetwork(XTrain, TTrain, layers(:,3), opts);

inofvald=[info.FinalValidationAccuracy, info2.FinalValidationAccuracy, info3.FinalValidationAccuracy]
[MAXVAl, Index_MAX_VAL]=max(inofvald)

disp(["Best archetectors of net " , num2str(Index_MAX_VAL)])

[net_1, info_1]=trainNetwork(XTrain, TTrain, layers(:,Index_MAX_VAL), opts);
[net_2, info_2]=trainNetwork(XTrain, TTrain, layers(:,Index_MAX_VAL), opts2);

if info_1.FinalValidationAccuracy>info_2.FinalValidationAccuracy
    test_net=net_1
else
    test_net=net_2
end


for i=1:5
   [net_2, info_2]=trainNetwork(XTrain, TTrain, layers(:,Index_MAX_VAL), opts2); 
end

% Load Test Data

imsetTest = imageSet('cifar10Test','recursive');

testNames = {imsetTest.Description};
XTest = zeros(32,32,3,sum([imsetTest.Count]),'uint8');
TTest = categorical(discretize((1:sum([imsetTest.Count]))',...
    [0,cumsum([imsetTest.Count])],'categorical',testNames));
j = 0;
tic;
for c = 1:length(imsetTest)
    for i = 1:imsetTest(c).Count
        XTest(:,:,:,i+j) = read(imsetTest(c),i);
    end
    j = j + imsetTest(c).Count;
end
toc;

% Run the network on the test set


% Alternative way using imageDataStore
imdsTest = imageDatastore(fullfile(pwd, 'cifar10Test'),...
     'IncludeSubfolders',true,'LabelSource','foldernames');
YTest = classify(test_net, imdsTest);

% Calculate the accuracy.
accuracy = sum(YTest == TTest)/numel(TTest)