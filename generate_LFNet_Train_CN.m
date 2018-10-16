%% Initialization
clear;
close all;

%% settings
length = 7;
scale = 2;
total_num = 100;
sample_num = 25;
stride = 45;
patch_size = 64;
batch_size = 64;
padding = 0;

DATA_folder = '../IMsF_RCNN/DATA/train/';
train_path = sprintf('LFNet_train_c1_C_s%d_l%d_f%d_p%d.mat',sample_num,length,scale,patch_size);


filepaths = dir(fullfile(DATA_folder,'*.png'));
first_img = imread(fullfile(DATA_folder,filepaths(1).name));
first_img = modcrop(first_img, scale);
[height,width,channel] = size(first_img);
clearvars filepaths first_img;

%% initialization
data = zeros(patch_size,patch_size,length,1);
label = zeros(patch_size,patch_size,length,1);

count = 1;

sample_order = randperm(total_num,sample_num);

for k = 1 : sample_num
    
    sample_indx = sample_order(k);

    fprintf('Folder %.3d is processing......\n',sample_indx);
    tic;
    
    n_folder = sprintf('%.3d/',sample_indx);
    data_folder = strcat(DATA_folder,n_folder);
    
    filepaths = dir(fullfile(data_folder,'*.png'));
    
    N = size(filepaths, 1);
    angular_size = sqrt(N);
    border = (angular_size-length)/2;
    
    all_gt = zeros(height,width,length);
    all_lr = zeros(height,width,length);
    
    for j = 1+border:2:angular_size-border
        for i = 1+border: angular_size-border
            data_filename = sprintf('view_%.2d_%.2d.png',i,j);
            img = imread(strcat(data_folder,data_filename));
            img = rgb2ycbcr(img);
            img = im2double(img(:, :, 1));
            img = modcrop(img, scale);
%             H = fspecial('gaussian',[3,3],2);
%             g_img = imfilter(img,H,'replicate');
            bicubic_img = imresize(imresize(img,1/scale),scale);
%             add_noise_img = imnoise(bicubic_img,'gaussian',0,0.0005);
%             subplot(121);imshow(bicubic_img);
%             subplot(122);imshow(img);
            all_lr(:,:,i-border) = bicubic_img;
            all_gt(:,:,i-border) = img;
            
%             if fb_flag > 0.5
%                 all_lr(:,:,j-border) = bicubic_img;
%                 all_gt(:,:,j-border) = img;
%             else
%              index = length+border+1-i;
%              all_lr(:,:,index) = bicubic_img;
%              all_gt(:,:,index) = img;
%             end
                
            %         figure;
            %         subplot(121);imshow(img);
            %         subplot(122);imshow(imresize(imresize(img,1/scale),scale));
        end
        
        for x = 1 : stride : height-patch_size+1
            for y = 1 :stride : width-patch_size+1
                
                subim_label = squeeze(all_gt(x+padding: x+padding+patch_size-1, y+padding : y+padding+patch_size-1,:));
                
                %             if numel(find(subim_label==0)) >= 0.9* numel(subim_label)
                %                 continue;
                %             end
                
                subim_input = squeeze(all_lr(x : x+patch_size-1, y : y+patch_size-1,:));
                
                data(:, :, :,count) = subim_input;
                
                label(:,:, :,count) = subim_label;
                
                count=count+1;
                
            end
        end
        
        
        
        clearvars all_gt all_lr;
        
    end
    
    toc
    
end

data = permute(data,[4 3 1 2]);
label = permute(label,[4 3 1 2]);

count = count-1;


order = randperm(count);
split = floor(0.95*count);
train_data = data(order(1:split),:, :, :);
train_label = label(order(1:split),:, :, :);
valid_data = data(order(split+1:end),:, :, :);
valid_label = label(order(split+1:end),:, :, :);

%% Size of data
disp('-----------------------');
disp('Train Data Size');
disp(num2str(size(train_data)));
disp('Train Label Size');
disp(num2str(size(train_label)));
disp('Valid Data Size');
disp(num2str(size(valid_data)));
disp('Valid Label Size');
disp(num2str(size(valid_label)));
disp('-----------------------');


%% writing to Matfile
disp('Saving to Mat.......');
save(train_path,'train_data','train_label','valid_data','valid_label','-v7.3');
disp('Done.');
