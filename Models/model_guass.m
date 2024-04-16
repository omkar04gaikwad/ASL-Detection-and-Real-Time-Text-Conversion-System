% Define the image data augmentation configuration
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection', true, ... % Horizontal flip
    'RandXScale', [0.8 1.2], ... % Random scaling for zoom
    'RandYScale', [0.8 1.2], ...
    'RandRotation', [-20 20], ... % Random rotation
    'RandXTranslation', [-10 10], ... % Random translation (shear-like effect)
    'RandYTranslation', [-10 10]);

% Specify training image datastore
trainDatastore = imageDatastore('dataset\preprocessedTrainingData\', ...
    'IncludeSubfolders', true, ...  % Assuming a folder structure by class
    'LabelSource', 'foldernames', ... % Labels from folder names
    'ReadFcn', @(x)imresize(imread(x),[128 128])); % Resize images

% Create augmented image datastore for training
trainAugimds = augmentedImageDatastore([128 128 1], trainDatastore, ...
    'DataAugmentation', imageAugmenter, ...
    'ColorPreprocessing', 'rgb2gray'); % Convert grayscale to RGB if needed
% Specify testing image datastore
testDatastore = imageDatastore('dataset\preprocessedTestingData\', ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames', ...
    'ReadFcn', @(x)imresize(imread(x),[128 128]));

% Create augmented image datastore for testing
testAugimds = augmentedImageDatastore([128 128 1], testDatastore, ...
    'ColorPreprocessing', 'rgb2gray'); % Keeping consistent with training

% Define the layers of the network
layers = [
    imageInputLayer([128 128 1], 'Name', 'input', 'Normalization', 'none')
    
    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv1')
    reluLayer('Name', 'relu1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool1')  % No 'Padding' needed for valid-like behavior
    
    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv2')
    reluLayer('Name', 'relu2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool2')  % Again, omit 'Padding'
    
    flattenLayer('Name', 'flatten')
    
    fullyConnectedLayer(128, 'Name', 'fc1')
    reluLayer('Name', 'relu3')
    dropoutLayer(0.4, 'Name', 'dropout1')
    
    fullyConnectedLayer(96, 'Name', 'fc2')
    reluLayer('Name', 'relu4')
    dropoutLayer(0.4, 'Name', 'dropout2')
    
    fullyConnectedLayer(64, 'Name', 'fc3')
    reluLayer('Name', 'relu5')
    
    fullyConnectedLayer(26, 'Name', 'fc4')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
];

% Define training options with validation data
options = trainingOptions('adam', ...
    'InitialLearnRate', 0.001, ...
    'MaxEpochs', 5, ...
    'MiniBatchSize', 10, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'ValidationData', testAugimds, ...  % Specify the validation data
    'ValidationFrequency', 30);

% Train the network
[trainedNet, trainInfo] = trainNetwork(trainAugimds, layers, options);

% Optionally, analyze the network
analyzeNetwork(trainedNet);


