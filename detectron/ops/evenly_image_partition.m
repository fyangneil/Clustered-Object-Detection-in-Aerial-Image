function EIP
%% crop an image to 6 image patches
im_path='/media/fanyang/C/data/visdrone/VisDrone2018-DET-val/images_coco_format_rename';
im_list=dir(fullfile(im_path,'*jpg'));
gt_path='/media/fanyang/C/data/visdrone/VisDrone2018-DET-val/annotations_coco_format_rename';
gt_list=dir(fullfile(gt_path,'*txt'));
im_output_path='/media/fanyang/C/data/visdrone/VisDrone2018-DET-val/eip_crop_image';
gt_output_path='/media/fanyang/C/data/visdrone/VisDrone2018-DET-val/eip_crop_gt';
width_num=3;
hieght_num=2;
padding=0;
if ~exist(im_output_path)
mkdir(im_output_path);
end
if ~exist(gt_output_path)
mkdir(gt_output_path);
end
for im_ind=1:length(im_list)
    im_ind
    im=imread(fullfile(im_path,im_list(im_ind).name));
    im_file=fullfile(im_path,im_list(im_ind).name);
    [h,w,~]=size(im);
    
    patch_h=round(h/hieght_num);
    patch_w=round(w/width_num);
    gt=importdata(fullfile(gt_path,gt_list(im_ind).name));
    gt_file=fullfile(gt_path,gt_list(im_ind).name);
    patch_left_list=[1:patch_w:w];
    patch_top_list=[1:patch_h:h];
    patch_h=patch_h+padding;
    patch_w=patch_w+padding;
    for patch_left_ind=1:width_num
        for patch_top_ind=1:hieght_num
            patch_left= patch_left_list(patch_left_ind);
            patch_top=patch_top_list(patch_top_ind);
            rect=[patch_left,patch_top,patch_w-1,patch_h-1];
            patch_right=patch_left+patch_w-1;
            patch_bottom=patch_top+patch_h-1;
            im_patch=imcrop(im,rect); 
            % show cropped image patch
            if 0
            figure(1)
            hold on
            imshow(im_patch);
            end 
            imwrite(im_patch,fullfile(im_output_path,[im_list(im_ind).name(1:end-4),'_',num2str(patch_left),'_',num2str(patch_top),'.jpg']));
            gt_output_file=fullfile(gt_output_path,[gt_list(im_ind).name(1:end-4),'_',num2str(patch_left),'_',num2str(patch_top),'.txt']);
            fileID = fopen(gt_output_file,'w');
            % create  new ground truth for each patch
            if isempty(gt)
                delete(im_file,gt_file); 
                continue;
            end
            for gt_ind=1:length(gt(:,1))
                
                bbox_left=gt(gt_ind,1);
                bbox_top=gt(gt_ind,2);
                bbox_w=gt(gt_ind,3);
                bbox_h=gt(gt_ind,4);
                bbox_right=bbox_left+bbox_w-1;
                bbox_bottom=bbox_top+bbox_h-1;
                area=bbox_w*bbox_h;
                score=gt(gt_ind,5);
                category=gt(gt_ind,6);
                truncation=gt(gt_ind,7);
                occlusion=gt(gt_ind,8);
                if bbox_left>=patch_right
                   continue;
                end
                
                if bbox_left<patch_left
                   bbox_left=patch_left;
                end
                
                if bbox_top>=patch_bottom
                   continue;
                end
                
                if bbox_top<patch_top
                   bbox_top=patch_top;
                end
                
                if bbox_right<=patch_left
                   continue;
                end
                
                if bbox_right>patch_right
                   bbox_right=patch_right;
                end
                
                if bbox_bottom<=patch_top
                   continue;
                end
                
                if bbox_bottom>patch_bottom
                   bbox_bottom=patch_bottom;
                end
                
                if bbox_right>bbox_left&&bbox_bottom>bbox_top
                   new_bbox_w=bbox_right-bbox_left+1;
                   new_bbox_h=bbox_bottom-bbox_top+1;
                   new_bbox_left=bbox_left-patch_left;
                   new_bbox_top=bbox_top-patch_top;
                   new_bbox=[new_bbox_left,new_bbox_top,new_bbox_w,new_bbox_h];
                   new_area=new_bbox_w*new_bbox_h;
                   if new_area/area>=0.5
                   %new_bbox_w>3||new_bbox_h>3
%                    rectangle('Position',new_bbox,'EdgeColor','r');

                   fprintf(fileID,'%d,%d,%d,%d,%d,%d,%d,%d\n',new_bbox_left,new_bbox_top,new_bbox_w,new_bbox_h,...
                                                               score,category,truncation,occlusion);                                         
                   end
                end          
            end
%             hold off
            fclose(fileID);     
        end
    end            
end

end
