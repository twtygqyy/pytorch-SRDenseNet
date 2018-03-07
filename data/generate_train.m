clear;
close all;
folder = 'path/to/imagenet/folder';

savepath = 'srdensenet_x4.h5';

%% scale factors
scale = 4;

size_label = 128;
size_input = size_label/scale;
stride = 128;

data = zeros(size_input/4, size_input/4, 1, 1);
label = zeros(size_label, size_label, 1, 1);

count = 0;
margain = 0;

%% generate data change image format to according to your image type
filepaths = [];
filepaths = [filepaths; dir(fullfile(folder, '*.JPEG'))];

length(filepaths)

for i = 1 : length(filepaths)
    image = imread(fullfile(folder,filepaths(i).name));
    if size(image,3)==3
        image = rgb2ycbcr(image);
        image = im2double(image);
        im_label = modcrop(image, scale);
        [hei,wid] = size(im_label);

        filepaths(i).name
        for x = 1 + margain : stride : hei-size_label+1 - margain
            for y = 1 + margain :stride : wid-size_label+1 - margain
                subim_label = im_label(x : x+size_input-1, y : y+size_input-1);
                subim_input = imresize(subim_label,1/scale(s),'bicubic');

                count=count+1;
                data(:, :, 1, count) = subim_input;
                label(:, :, 1, count) = subim_label;
            end
        end
    end
end

order = randperm(count);
data = data(:, :, 1, order);
label = label(:, :, 1, order);

%% writing to HDF5
chunksz = 64;
created_flag = false;
totalct = 0;

for batchno = 1:floor(count/chunksz)
    batchno
    last_read=(batchno-1)*chunksz;
    batchdata = data(:,:,:,last_read+1:last_read+chunksz); 
    batchlabs = label(:,:,:,last_read+1:last_read+chunksz);
    startloc = struct('dat',[1,1,1,totalct+1], 'lab', [1,1,1,totalct+1]);
    curr_dat_sz = store2hdf5(savepath, batchdata, batchlabs, ~created_flag, startloc, chunksz); 
    created_flag = true;
    totalct = curr_dat_sz(end);
end

h5disp(savepath);
