function copy_syn_images

id = '04004475';
src_root = '/cvgl/u/cwind/KDE/RenderForCNN-master/data/syn_images';
dst_root = '/capri5/Projects/ImageNet3D/Syn_Images';

dst_dir = fullfile(dst_root, id);
if exist(dst_dir, 'dir') == 0
    mkdir(dst_dir);
end

filename = fullfile(dst_root, [id '.txt']);
fid = fopen(filename, 'w');

src_dir = fullfile(src_root, id);
src_files = dir(src_dir);
N = numel(src_files);

count = 0;
for i = 1:N
    name = src_files(i).name;
    if src_files(i).isdir == 1 && strcmp(name, '.') == 0 && strcmp(name, '..') == 0
        image_files = dir(fullfile(src_root, id, name, '*.png'));
        n = numel(image_files);
        if n > 0
            count = count + 1;
        end
        for j = 1:n
            img_name = image_files(j).name;
            src_file = fullfile(src_root, id, name, img_name);
            dst_file = fullfile(dst_dir, img_name);
            copyfile(src_file, dst_file);
            fprintf(fid, '%s %d\n', img_name, count-1);
            disp(img_name);
        end
    end
end

fclose(fid);