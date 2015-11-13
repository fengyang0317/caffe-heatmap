% This file uses a FLIC trained model and applies it to a video sequence from Poses in the Wild
%
% Download the model:
%    wget http://tomas.pfister.fi/models/caffe-heatmap-flic.caffemodel -P ../../models/heatmap-flic-fusion/

% Options

opt.visualise = true;		% Visualise predictions?
opt.useGPU = false; 			% Run on GPU
opt.dims = [256 256]; 		% Input dimensions (needs to match matlab.txt)
opt.numJoints = 7; 			% Number of joints
opt.layerName = 'conv5_fusion'; % Output layer name
opt.modelDefFile = '../../models/heatmap-flic-fusion/matlab.prototxt'; % Model definition
opt.modelFile = '../../models/heatmap-flic-fusion/caffe-heatmap-flic.caffemodel'; % Model weights
%opt.modelFile = '../../snapshots/heatmap_train_iter_60000.caffemodel';

% Add caffe matlab into path
addpath('../')

% Image input directory
opt.inputDir = 'sample_images/';
opt.inputDir = '~/lab/pose/FLIC/images256/';

% Create image file list
%imInds = 1:29;
%for ind = 1:numel(imInds); files{ind} = [num2str(imInds(ind)) '.png']; end
imgs = dir('~/lab/pose/FLIC/images256/*.jpg');
for i = 1:length(imgs)
    files{i} = imgs(i).name;
end

% Apply network
joints = applyNet(files, opt)