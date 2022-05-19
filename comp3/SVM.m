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
if (~exist('SVMModelsRBF.mat', 'file'))||(~exist('SVMModelsRBF.mat', 'file'))
trainNames = {imsetTrain.Description};
XTrain = zeros( sum([imsetTrain.Count]), 1024);

TTrain = discretize((1:sum([imsetTrain.Count])),...
    [0,cumsum([imsetTrain.Count])],trainNames);

j=0
tic;
for c = 1:length(imsetTrain)
    for i = 1:imsetTrain(c).Count
        
        inputImg = im2double(read(imsetTrain(c),i));
        inputImg = rgb2gray(inputImg);
        inputImg = imresize(inputImg, [32 32]);
        inputImg = reshape (inputImg', 1, size(inputImg,1)*size(inputImg,2));
        XTrain(i+j,: ) =inputImg;
       
    end
    j = j + imsetTrain(c).Count
end

toc;

end

% Define a SVM architecture
if ~exist('SVMModelsRBF.mat', 'file')
SVMModelsRBF = cell(10,1);


classes = unique(TTrain);
rng(1); % For reproducibility
%ct = cvpartition(50000,'KFold',5000);
%opts = struct('CVPartition',ct,'AcquisitionFunctionName','expected-improvement-plus');
tic
for j = 1:numel(classes)
    indx = strcmp(TTrain,classes(j)); % Create binary classes for each classifier
    disp(['Start SVM train number ', num2str(j)]);
    SVMModelsRBF{j} = fitcsvm(XTrain,indx,'ClassNames',[false true],'Standardize',true, ...
        'KernelFunction','rbf', 'KernelScale','auto');
    disp(['Stop SVM model number ', num2str(j)]);
    
end
toc
save('SVMModelsRBF', 'SVMModelsRBF')
end

fclose('all')

if ~exist('SVMModelsPol.mat', 'file')
SVMModelsPol = cell(10,1);


classes = unique(TTrain);
rng(1); % For reproducibility
%ct = cvpartition(50000,'KFold',5000);
%opts = struct('CVPartition',ct,'AcquisitionFunctionName','expected-improvement-plus');
tic
for j = 1:numel(classes)
    indx = strcmp(TTrain,classes(j)); % Create binary classes for each classifier
    disp(['Start SVM train number ', num2str(j)]);
    SVMModelsPol{j} = fitcsvm(XTrain,indx,'ClassNames',[false true],'Standardize',true, ... 
        'KernelFunction', 'polynomial');
    disp(['Stop SVM model number ', num2str(j)]);
    
end
toc
save('SVMModelsPol', 'SVMModelsPol')


end


fclose('all')

% Cross validation

if ~exist('CVSVMModelsRBF.mat', 'file')
CVSVMModelsRBF = cell(10,1);
tic
load('SVMModelsRBF.mat', 'SVMModelsRBF')
fclose('all')
tic
for j = 1:10
    
    disp(['Start cross val SVM  ', num2str(j)]);
    CVSVMModelsRBF{j} = crossval(SVMModelsRBF{j}, 'Kfold', 2)
    disp(['Stop cross val SVM model number ', num2str(j)]);
    
end
toc
save('CVSVMModelsRBF', 'CVSVMModelsRBF')
end


if ~exist('CVSVMModelsPol.mat', 'file')
CVSVMModelsPol = cell(10,1);
tic
load('SVMModelsPol.mat', 'SVMModelsPol')
fclose('all')
tic
for j = 1:10
    
    disp(['Start cross val SVM  ', num2str(j)]);
    CVSVMModelsPol{j} = crossval(SVMModelsPol{j}, 'Kfold', 2)
    disp(['Stop cross val SVM model number ', num2str(j)]);
    
end
toc
save('CVSVMModelsPol', 'CVSVMModelsPol')
end


% Load Test Data

imsetTest = imageSet('cifar10Test','recursive');

testNames = {imsetTest.Description};
XTest = zeros(sum([imsetTest.Count]), 1024);
TTest = discretize((1:sum([imsetTest.Count]))',...
    [0,cumsum([imsetTest.Count])],testNames);
j = 0;
tic;
for c = 1:length(imsetTest)
    for i = 1:imsetTest(c).Count
        
        inputImg = im2double(read(imsetTest(c),i));
        inputImg = rgb2gray(inputImg);
        inputImg = imresize(inputImg, [32 32]);
        inputImg = reshape (inputImg', 1, size(inputImg,1)*size(inputImg,2));
        XTest(i+j,:) = inputImg;
    end
    j = j + imsetTest(c).Count;
end
toc;

% Run the network on the test set
if ~exist('RBF.mat', 'file')
    
classes = unique(TTest);    
    
tic
load('CVSVMModelsRBF.mat', 'CVSVMModelsRBF')
fclose('all')
tic
RBF=cell(10,1);   
N = size(XTest,1);
Scores = zeros(N,numel(classes));

for j = 1:numel(classes)
    indx = strcmp(TTest,classes(j));
    [labels,score] = predict(CVSVMModelsRBF{j}.Trained{1,1},XTest);
     RBF{j}=confusionmat(indx, labels);% Second column contains positive-class scores
end

save('RBF.mat', 'RBF')

end



% Run the network on the test set
if ~exist('Pol.mat', 'file')
    
classes = unique(TTest);    
    
tic
load('CVSVMModelsPol.mat', 'CVSVMModelsPol')
fclose('all')
tic
Pol=cell(10,1);   
N = size(XTest,1);
Scores = zeros(N,numel(classes));

for j = 1:numel(classes)
    indx = strcmp(TTest,classes(j));
    [labels,score] = predict(CVSVMModelsPol{j}.Trained{1,1},XTest);
     Pol{j}=confusionmat(indx, labels);% Second column contains positive-class scores
end

save('Pol.mat', 'Pol')

end
% Calculate the accuracy.
