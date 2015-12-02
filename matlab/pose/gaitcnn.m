clear all

opt.visualise = true;		% Visualise predictions?
opt.useGPU = false; 			% Run on GPU
opt.dims = [256 256]; 		% Input dimensions (needs to match matlab.txt)
opt.numJoints = 7; 			% Number of joints
opt.layerName = 'conv5_fusion'; % Output layer name
%opt.layerName = 'conv8';
opt.modelDefFile = '../../models/heatmap-flic-fusion/matlab.prototxt'; % Model definition
opt.modelFile = '../../models/heatmap-flic-fusion/caffe-heatmap-flic.caffemodel'; % Model weights
%opt.modelFile = '../../snapshots/heatmap_train_iter_120000.caffemodel';

% Add caffe matlab into path
addpath('../')

% Image input directory
%opt.inputDir = 'sample_images/';
%opt.inputDir = '~/lab/pose/FLIC/images256/';
opt.inputDir = '/home/yfeng23/lab/dataset/gait/DatasetB/';

files = importdata([opt.inputDir 'croplist']);

opt.numFiles = numel(files);
net = initCaffe(opt); 

for ind = 1:opt.numFiles
	imFile = files{ind};
    [a, b, c]=fileparts(imFile);
	fprintf('file: %s\n', imFile);
    
    img = imread([opt.inputDir imFile]);
    img = single(img);
    if size(img,1)>size(img,2)
        img = padarray(img,[0,floor(size(img,1)-size(img,2))/2,0]);
    end
    img = imresize(img, opt.dims);
    input_data = permute(img, [2 1 3]);

    net.forward({input_data});
    features = net.blobs(opt.layerName).get_data();
    save([opt.inputDir 'cnn/' b],'features');
end