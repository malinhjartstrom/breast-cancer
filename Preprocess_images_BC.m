%% Import images
%myFolder = 'C:\Users\hjart\Documents\LTH\Master_thesis\'; %Lenovo
%filePattern = fullfile(myFolder, '*.png'); %Lenovo
clear;

myFolder = '/scratch/bob/malin_hj/decImages/'; %Bob
filePattern = fullfile(myFolder, '*.tif'); %Bob

files = dir(filePattern);
left_idx = 1;
right_idx = 1;
%images_left = {};
%left_files = {};
%images_right = {};
%right_files = {};

remove_right = [2 13 19 40 45 50 51 53 59 62 63 64 70 73 83 90 98 ...
    107 108 113 115 117 131 136 140 143 147 149 153 157 167 172 173 175 177 191 196 199 ...
    205 220 233 237 252 271 272 282 283 285 286 287 288 290 294 297 ...
    314 316 318 321 322 327 332 337 339 345 351 356 359 367 369 376 380 382 384 398 ...
    401 405 420 427 429 431 438 439 440 449 451 456 462 464 472 474 478 481 483 491 ...
    502 505 507 511 512 521 527 530 534 537 539 543 578 589 ... 
    603 621 629 631 635 639 645 649 651 653 654 657 662 664 667 674 682 685 692 698 ...
    700 718 724 730 733 735 741 747 749 753 759 761 763 770 773 778 793 796 798 ...
    800 802 809 812 816 819 823 827 832 838 841 844 847 850 868 875 880 885 ...
    907 912 914 916 919 922 925 927 941 944 950 957 960];

remove_left = [8 12 31 38 64 74 84 92 94 99 ...
    104 109 155 159 167 177 192 194 198 ...
    202 209 221 223 231 248 254 279 285 287 292 294 ...
    301 303 312 314 315 318 323 331 343 347 351 353 355 364 370 372 376 380 385 387 394 ...
    405 408 410 413 419 421 423 435 432 441 444 447 449 451 466 471 475 482 493 495 498 ...
    500 508 521 523 528 530 532 540 563 593 ...
    607 615 624 628 644 647 650 654 656 660 668 681 683 687 688 696 ...
    702 708 710 712 723 725 731 733 736 744 753 760 770 783 786 789 792 796 ...
    803 806 811 812 815 820 827 833 844 848 852 856 876 889 891 ...
    906 909 912 914 926 934];

for k = 1 : length(files)
%     if contains(files(k).name, 'LCC')
%         if ~ismember(left_idx, remove_left)
%             fullFileName = fullfile(files(k).folder, files(k).name);
%             fprintf(1, 'Now reading %s\n', fullFileName);
%             image_left = imread(fullFileName);
%             left_file = files(k);
%             left_idx = left_idx + 1;
%             left_image_norm = double(image_left)./double(max(image_left)); %normalizing
%             level = graythresh(left_image_norm); %Otsu
%             BW_L = imbinarize(left_image_norm, level); % Binarize image
%             [left_cropped_width, bin_l] = cropOtsuSingle(left_image_norm, BW_L); % crop in width
%             left_cropped_final = cropOtsuHeight2single(left_cropped_width, bin_l); % Crop in height
%             left_re = imresize(left_cropped_final, [600 400]); % resizing image
%             saveTIFsingle(left_re, left_file); % Saving as TIF
%         end
%     elseif contains(files(k).name, 'RCC')
%         if ~ismember(right_idx, remove_right) % if the index is not in the array of indices to remove
%             fullFileName = fullfile(files(k).folder, files(k).name);
%             fprintf(1, 'Now reading %s\n', fullFileName);
%             image_right = imread(fullFileName);
%             right_file = files(k);
%             right_idx = right_idx + 1;
%             right_images_mirrored = flip(image_right,2); %flipping the right images
%             right_image_norm = double(right_images_mirrored)./double(max(right_images_mirrored)); %normalizing
%             lvl = graythresh(right_image_norm); % Otsu
%             BW_R = imbinarize(right_image_norm, lvl); %Binarize image
%             [right_cropped_width, bin_r] = cropOtsuSingle(right_image_norm, BW_R); % Crop in width
%             right_cropped_final = cropOtsuHeight2single(right_cropped_width, bin_r); % Crop in height
%             right_re = imresize(right_cropped_final, [600 400]); %resizing image
%             saveTIFsingle(right_re, right_file); %saving as TIF
%         end
    if contains(files(k).name, 'Lcranio') %elseif contains(files(k).name, 'Lcranio')
        fullFileName = fullfile(files(k).folder, files(k).name);
        fprintf(1, 'Now reading %s\n', fullFileName);
        image_left = imread(fullFileName);
        left_file = files(k);
        left_idx = left_idx + 1;
        left_image_norm = double(image_left)./double(max(image_left)); %normalizing
        level = graythresh(left_image_norm); %Otsu
        BW_L = imbinarize(left_image_norm, level); % Binarize image
        BW_transl = 1-BW_L; % transpose image
        [left_cropped_width, bin_l] = cropOtsuSingle(left_image_norm, BW_transl); % crop in width
        left_cropped_final = cropOtsuHeight2single(left_cropped_width, bin_l); % Crop in height
        left_re = imresize(left_cropped_final, [600 400]); % resizing image
        imshow(left_re, 'InitialMagnification', 'fit')
        pause(5)
        %saveTIFLcranio(left_re, left_file); % Saving as TIF     
    elseif contains(files(k).name, 'Rcranio')
        fullFileName = fullfile(files(k).folder, files(k).name);
        fprintf(1, 'Now reading %s\n', fullFileName);
        image_right = imread(fullFileName);
        right_file = files(k);
        right_idx = right_idx + 1;
        right_image_norm = double(image_right)./double(max(image_right)); %normalizing
        level = graythresh(right_image_norm); %Otsu
        BW_R = imbinarize(right_image_norm, level); % Binarize image
        BW_transr = 1-BW_R;% transpose image
        [right_cropped_width, bin_r] = cropOtsuSingle(right_image_norm, BW_transr); % crop in width
        right_cropped_final = cropOtsuHeight2(right_cropped_width, bin_r); % Crop in height
        right_re = imresize(right_cropped_final, [600 400]); % resizing image
        imshow(right_re, 'InitialMagnification', 'fit')
        pause(5)
        %saveTIFRcranio(right_re, right_file); % Saving as TIF     
    end
end

%% Test Maren 9. Mars 
% testing left images
%left_idx = 0;
right_idx = 0;
for k = 1:length(files)
   if contains(files(k).name, 'Rcranio')
    right_idx = right_idx + 1;
    fullFileName = fullfile(files(k).folder, files(k).name);
    fprintf(1, 'Now reading %s\n', fullFileName);
    image_right = imread(fullFileName);
    image_right_norm = double(image_right)./double(max(image_right)); %norm(image_left);
    right_idx
    imshow(image_right_norm, 'InitialMagnification', 'fit')
    pause(5)
   end
end



%%
for k = 1:length(images_right)
    right_images_mirrored{k} = flip(images_right{k}, 2); % mirror images
end
'Images mirrored.'

%% Normalize images (to be able to view them)
l_images = norm(images_left);
'Left images normalized.'
%%
r_images = norm(images_right);
'Right images normalized.'

%% Show images as subplots
for k = 1:length(l_images) %left images
    subplot(1,length(l_images), k)
    imshow(l_images{k})
end

%% Otsu segmentation
for k = 1: length(right_images_mirrored)
    level = graythresh(right_images_mirrored{k});
    BW_R{k} = imbinarize(right_images_mirrored{k}, level);
end

%%
for k = 1: length(l_images)
    level = graythresh(l_images{k});
    BW_L{k} = imbinarize(l_images{k}, level);
end

imshowpair(l_images{1}, BW_L{1}, 'montage');

%% Crop images in width (Bob)
[crop_left, otsu_l] = cropOtsu(l_images, BW_L);
%[crop_right, otsu_r] = cropOtsu(right_images_mirrored, BW_R);

figure
subplot(131)
imshow(crop_left{1})
subplot(132)
imshow(otsu_l{1})
subplot(133)
imshow(l_images{1})
%% Crop images in height (Bob)
L_cropIm = cropOtsuHeight2(crop_left, otsu_l);
%R_cropIm = cropOtsuHeight2(crop_right, otsu_r);

for k = 1: length(L_cropIm)
    figure
    subplot(131)
    imshow(L_cropIm{k});
    subplot(132)
    imshow(l_images{k});
    subplot(133)
    imshow(otsu_l{k});
end
 %% Make images 2D (only for Lenovo)
% for k = 1:length(left_images)
%     L_im_2D{k} = rgb2gray(left_images{k});
% end
% 
% for k = 1:length(right_images)
%     R_im_2D{k} = rgb2gray(right_images_mirrored{k});
% end


%%
imshow(L_croppedImages{1,17})

%L_croppedImages{1,6} contains white to the right
%L_croppedImages{1,8} is a zoomed image
%L_croppedImages{1,12} is a zoomed image
 
%% Resize images
for k = 1: length(L_croppedImages)
    L_resized{k} = imresize(L_croppedImages{k}, [600, 400]);
end 
for k = 1: length(R_croppedImages)
    R_resized{k} = imresize(R_croppedImages{k}, [600, 400]);
end

%% Control size of images
size(L_resized{1,2})
size(R_resized{1,2})

figure
imshow(L_resized{1,2})

%% Save images as TIF
saveTIF(L_resized, left_files);
saveTIF(R_resized, right_files);

%% Functions

% Set images between [0,1]
function [mammograms] = norm(images)
    for k = 1 : length(images)
        image = images{k};
        mammograms{k} = double(image)./double(max(image));
        k
    end
end

% Make background darker
function L = setBlack(im, below_level)
for k = 1 : length(im)
    image = im{k};
    idx = find(abs(image)<below_level);
    image(idx) = 0;
    L{k} = image;
end
end

% Make white parts whiter
function lighter = setWhite(im, above_level)
    for k = 1:length(im)
        image = im{k};
        idx = find(abs(image)>above_level);
        image(idx) = 1;
        lighter{k} = image;
    end
end

% Find index where images stops being white at the side and crop
function [newImages] = cropBackground(original_images, whitened_images)
    for k = 1:length(whitened_images) % no of images
        height = size(whitened_images{1,k},1);
        limit = height-5;
        % For images with a mark in the top right corner
        original_images{1,k}(1:ceil(0.15*height),ceil(size(original_images{1,k},2)*0.6):end) = 0;
        imshow(original_images{1,k})
        % Crop images
        for width = ceil(size(whitened_images{1,k},2)*0.75):-1: 1 % no of columns (true width)
            if sum(whitened_images{1,k}(:,width)) < limit
                newImages{k} = imcrop(original_images{1,k}, [0 0 width height]);
                break;
            else 
                newImages{k} = original_images{k};
            end
        end
    end
end

% Crop in width after Otsu
function [im, otsu] = cropOtsu(originals, otsu_im)
    im = {};
    otsu = {};
    for k = 1:length(otsu_im)
        height=size(otsu_im{1,k},1); 
        for width = 100:size(otsu_im{1,k},2) % width = 1:size
            if sum(otsu_im{1,k}(:,width)) > height - 100 % == height
                im{k} = imcrop(originals{1,k}, [0 0 width-1 height]);
                otsu{k} = imcrop(otsu_im{1,k}, [0 0 width-1 height]);
                break;
            elseif sum(otsu_im{1,k}(ceil(0.25*height):ceil(0.75*height),width)) == 0
                im{k} = imcrop(originals{1,k}, [0 0 width height]);
                otsu{k} = imcrop(otsu_im{1,k}, [0 0 width height]);
                break;
            else
                im{k} = originals{k};
                otsu{k} = otsu_im{k};
            end
        end
    end
end

% Crop in width after Otsu
function [im, otsu] = cropOtsuSingle(original, otsu_im)
    height=size(otsu_im,1); 
    for width = 100:size(otsu_im,2) % width = 1:size
        if sum(otsu_im(:,width)) > height - 100 % == height
            im = imcrop(original, [0 0 width-1 height]);
            otsu = imcrop(otsu_im, [0 0 width-1 height]);
            break;
        elseif sum(otsu_im(ceil(0.25*height):ceil(0.75*height),width)) == 0
            im = imcrop(original, [0 0 width height]);
            otsu = imcrop(otsu_im, [0 0 width height]);
            break;
        else
            im = original;
            otsu = otsu_im;
        end
    end
end

function im = cropOtsuHeight2single(originals, otsu_im)
    mid = size(otsu_im,1)/2; % middle of image in height
    width = size(otsu_im,2);
    height = size(otsu_im,1);
    ymin = 0;
    ymax = height;
    for row = mid:height-1
        if sum(otsu_im(row,:)) < 60
           ymax = row + 50;
           break;
        end
    end
    for row = mid:-1:1
        if sum(otsu_im(row,:)) < 60
            ymin = row-50;
            break;
        end
    end
    new_height = ymax-ymin;
    im = imcrop(originals, [0 ymin width new_height]);
end

function [im] = cropOtsuHeight2(originals, otsu_im)
    im = {};
    for k = 1 : length(otsu_im)
        mid = size(otsu_im{1,k},1)/2; % middle of image in height
        width = size(otsu_im{1,k},2);
        height = size(otsu_im{1,k},1);
        ymin = 0;
        ymax = height;
        for row = mid:height-1
            if sum(otsu_im{1,k}(row,:)) < 60
               ymax = row + 50;
               break;
            end
        end
        for row = mid:-1:1
            if sum(otsu_im{1,k}(row,:)) < 60
                ymin = row-50;
                break;
            end
        end
        new_height = ymax-ymin;
        im{k} = imcrop(originals{1,k}, [0 ymin width new_height]);
    end
end

function [im] = cropHeightOtsu(originals, otsu_im)
    im = [];
    for k = 1:length(otsu_im)
        k
        width = size(otsu_im{1,k},2)
        whiteness_below = sum(otsu_im{1,k}(1,:)) % whiteness in the bottom of the image
        whiteness_above = sum(otsu_im{1,k}(size(otsu_im{1,k},1),:)) % whiteness in the top of the image
        for ymin = 2:size(otsu_im{1,k},1) %bottom up
            whiteness = sum(otsu_im{1,k}(ymin,:));
            if whiteness > whiteness_below
                ymin_output = ymin-1;
                break;
            elseif whiteness_below == 0 || whiteness_below > whiteness
                whiteness_below = whiteness; %moving one step up
            end
        end
        for height = size(otsu_im{1,k},1)-1:-1:1 %top down
            whiteness = sum(otsu_im{1,k}(height,:));
            if whiteness > whiteness_above
                height_output = height -1;
                break;
            elseif whiteness_above == 0 || whiteness_above > whiteness
                whiteness_above = whiteness; %moving one step down
            end
        end
        ymin_output
        h = height_output - ymin_output
        starting_height = size(otsu_im{1,k},1)
        im{k} = imcrop(originals{1,k}, [0 ymin_output width h]);
    end
end
        

% Find index to crop the image in width
function [cropCol] = cropWidth(im, limit)
    cropCol = [];
    for k = 1:length(im)
        for col = 50: size(im{1,k},2) % number of columns for each image. Starting from 50 if there is black in the start.
            if sum(im{1,k}(:,col)) < limit
                cropCol{k} = col;
                break;
            else
                cropCol{k} = 0;
            end
        end
    end
end

% Crop the image in height
function croppedImages = cropHeight(original_image, blackened_image, limit) % limit is when there are enough light pixels that it's not background anymore
    croppedImages = [];
    for k = 1:length(original_image)
        width = size(original_image{1,k},2);
        height = size(original_image{1,k},1);
        blackened_image{1,k} = bwareaopen(blackened_image{1,k}, 300); %remove small objects
        for row = 1: size(original_image{1,k},1)-50 % -50 to not go all the way ot the bottom
            if sum(blackened_image{1,k}(row,:)) > limit
                ymin = row-1
                break;
            else
                ymin = 0
            end
        end
        for row = size(original_image{1,k},1):-1: 50 % 50 to not go all the way ot the top
            sum(original_image{1,k}(row,:))
            if sum(blackened_image{1,k}(row,:)) > limit
                h = row + 1
                break;
            else
                h = height
            end
        end
        croppedImages{k} = imcrop(original_image{1,k}, [0 ymin, width, h]);
    end
end


% Crop images at the width found in the cropIdx function
function [croppedImages] = cropImages(im, ymin, width, height)
    for k = 1 : length(im)
        w = size(im{1,k},2);
        h = size(im{1,k},1);
        if ymin{k} == 0 && width{k} == 0 && height{k} == 0
            croppedImages{k} = im{1,k};
        elseif ymin{k} == 0 && width{k} == 0 && height{k} ~= 0
            croppedImages{k} = imcrop(im{1,k}, [0, 0, w, height{k}]);
        elseif ymin{k} == 0 && width{k} ~= 0 && height{k} == 0
            croppedImages{k} = imcrop(im{1,k}, [0, 0, width{k}, h]);
        elseif ymin{k} == 0 && width{k} ~= 0 && height{k} ~= 0
            croppedImages{k} = imcrop(im{1,k}, [0, 0, width{k}, height{k}]);
        elseif ymin{k} ~= 0 && width{k} == 0 && height{k} == 0
            croppedImages{k} = imcrop(im{1,k}, [0, ymin{k}, w, h-ymin{k}]);     
        elseif ymin{k} ~= 0 && width{k} == 0 && height{k} ~= 0
            croppedImages{k} = imcrop(im{1,k}, [0, ymin{k}, w, height{k}-ymin{k}]);
        elseif ymin{k} ~= 0 && width{k} ~= 0 && height{k} == 0
            croppedImages{k} = imcrop(im{1,k}, [0, ymin{k}, width{k}, h-ymin{k}]);
        else
            croppedImages{k} = imcrop(im{1,k}, [0, ymin{k}, width{k}, height{k}-ymin{k}]);            
        end
    end
end

% Resize images
function images = resize(im, width, height)
    for k = 1: length(im)
        images{k} = imresize(im{k},[height width]);
    end
end

% Save images as png files
function savePNG(im, info)
    %savePath = 'C:\Users\hjart\Documents\LTH\'; %Lenovo
    savePath = '/scratch/bob/malin_hj/decImages/'; %Bob
    for k = 1 : length(im)
        str = info(k).name; % name of image
        fileName = strtok(str, '.'); % remove old file format
        fileNamePNG = append(fileName, '.png'); % add PNG as new file format
        fullFileName = fullfile(savePath, fileNamePNG) % path + filename as full filename
        imwrite(im{k}, fullFileName, 'png'); % save image
    end
end

function saveTIF(im, info)
    savePath = '/scratch/bob/malin_hj/decImages/revImages';
    for k = 1 : length(im)
        str = info(k).name;
        fileName = strtok(str, '.');
        fileNameTIF = append(fileName, '_rev.tif');
        fullFileName = fullfile(savePath, fileNameTIF);
        imwrite(im{k}, fullFileName, 'tif');
    end
end

function saveTIFsingle(im, info)
    savePath = '/scratch/bob/malin_hj/decImages/revImages'; %/????
    str = info.name;
    fileName = strtok(str, '.');
    fileNameTIF = append(fileName, '_rev.tif');
    fullFileName = fullfile(savePath, fileNameTIF);
    imwrite(im, fullFileName, 'tif');
end

function saveTIFLcranio(im, info)
    savePath = '/scratch/bob/malin_hj/decImages/revImages'; %/????
    str = info.name;
    old = 'Lcranio_caudal_';
    new = 'LCC';
    newStr = strrep(str, old, new);
    fileName = strtok(newStr, '.');
    fileNameTIF = append(fileName, '_rev.tif');
    fullFileName = fullfile(savePath, fileNameTIF);
    imwrite(im, fullFileName, 'tif');
end

function saveTIFRcranio(im, info)
    savePath = '/scratch/bob/malin_hj/decImages/revImages'; %/????
    str = info.name;
    old = 'Rcranio_caudal_';
    new = 'RCC';
    newStr = strrep(str, old, new);
    fileName = strtok(newStr, '.');
    fileNameTIF = append(fileName, '_rev.tif');
    fullFileName = fullfile(savePath, fileNameTIF);
    imwrite(im, fullFileName, 'tif');
end
