function fuse_global_cluster_detections
%% collect detections on cropped images to generate final detections
img_path='/media/fanyang/C/data/visdrone/VisDrone2018-DET-val/images_coco_format_rename';
img_crop_path='/media/fanyang/C/data/visdrone/VisDrone2018-DET-val/cluster_crop_image_CPNet';
cluster_det_file='/media/fanyang/C/data/visdrone/VisDrone2018-DET-val/det_result/cluster_crop_image_CPNet_det_result/bbox_coco_2014_val_results.json';
global_det_file='/media/fanyang/C/data/visdrone/VisDrone2018-DET-val/det_result/coarse_det_result/bbox_coco_2014_val_results.json';
cluster_gt_file='/media/fanyang/C/data/visdrone/VisDrone2018-DET-val/coco_format_annotations/cluster_CPNet.json';
global_gt_file='/media/fanyang/C/data/visdrone/VisDrone2018-DET-val/coco_format_annotations/instances_val2014.json';
output_file='/media/fanyang/C/data/visdrone/VisDrone2018-DET-val/det_result/final_det_result/detections.json';
cocoGt_global=CocoApi(global_gt_file);
cocoDt_global=cocoGt_global.loadRes(global_det_file);

cocoGt_cluster=CocoApi(cluster_gt_file);
cocoDt_cluster=cocoGt_cluster.loadRes(cluster_det_file);
img_list=dir(fullfile(img_path,'*.jpg'));
img_crop_list=dir(fullfile(img_crop_path,'*.jpg'));
det_in_img=[];
det_original_in_img=[];
if 1
for i=1:length(img_list)
    det_in_img_i=[];
    image_id=i
    img_name=img_list(i).name(1:end-4);
    img_crop_ids=find(contains({img_crop_list.name},img_name));
    det_original_in_img_i=cocoDt_global.data.annotations(find([cocoDt_global.data.annotations.image_id]==image_id));
    for img_crop_i=1:length(img_crop_ids)
        img_crop_id=img_crop_ids(img_crop_i);
        img_crop_name=img_crop_list(img_crop_id).name;
        img_crop_coco_name=img_crop_list(img_crop_id).name;
        %get the xy of cropped region
        img_crop_name_splits=strsplit(img_crop_name,{'_','.'});
        crop_xy=[str2num(img_crop_name_splits{4}),str2num(img_crop_name_splits{5})];
        %get the detections in the cropped regions
        img_crop_coco_id=find(strcmp({cocoDt_cluster.data.images.file_name},img_crop_coco_name));
        det_ids=find([cocoDt_cluster.data.annotations.image_id]==img_crop_coco_id);
        det_in_crop=cocoDt_cluster.data.annotations(det_ids);
        %show detection in crop region
        if 0 
            img_crop_file=fullfile(img_crop_path,img_crop_coco_name);
            img_crop=imread(img_crop_file);
            imshow(img_crop);
            hold on
            for det_i=1:length(det_in_crop)
                rect=det_in_crop(det_i).bbox(:);
                rectangle('Position',rect,'EdgeColor','r');
            end
            pause(0.3)
            hold off
        end
        
        %%process truncated detections in crop
            if 1
                dis_to_boundary=5; 
                img_crop_file=fullfile(img_crop_path,img_crop_coco_name);
                img_crop=imread(img_crop_file);
                [h,w,~]=size(img_crop);
                det_i_keep=[];
                for det_i=1:length(det_in_crop)
                    rect=det_in_crop(det_i).bbox(:);
                    x1=rect(1);y1=rect(2);bbox_w=rect(3);bbox_h=rect(4);
                    x2=x1+bbox_w-1;y2=y1+bbox_h-1;
                    category_id=det_in_crop(det_i).category_id;
                    %screen the detection of bus (catogery 3)
%                     if category_id==3
%                         continue;
%                     end
                    if x1>=dis_to_boundary&&y1>=dis_to_boundary&&x2<=w-dis_to_boundary&&y2<=h-dis_to_boundary
                        det_i_keep=[det_i_keep;det_i];
                       
                    end


                end
                det_in_crop=det_in_crop(det_i_keep);
            end
    
        
        %show detection in crop region after processing truncated detections
        if 0
            img_crop_file=fullfile(img_crop_path,img_crop_coco_name);
            img_crop=imread(img_crop_file);
            imshow(img_crop);
            hold on
            for det_i=1:length(det_in_crop)
                rect=det_in_crop(det_i).bbox(:);
                rectangle('Position',rect,'EdgeColor','b');
            end
            pause(0.3)
            hold off
        end
        %get the detections in the original image
        for det_i=1:length(det_in_crop)
            det_in_crop(det_i).bbox(1)=det_in_crop(det_i).bbox(1)+crop_xy(1)-1;
            det_in_crop(det_i).bbox(2)=det_in_crop(det_i).bbox(2)+crop_xy(2)-1;
            det_in_crop(det_i).image_id=image_id;
        end
        det_in_img=[det_in_img,det_in_crop];
        det_in_img_i=[det_in_img_i,det_in_crop];
        %keep the detection which are not in crop regions 
        if 1
            dis_to_boundary=5;
            [h,w,~]=size(img_crop);
            crop_x1=crop_xy(1);crop_y1=crop_xy(2);crop_x2=crop_x1+w-1;crop_y2=crop_y1+h-1;
            det_i_keep=[];
            for det_i=1:length(det_original_in_img_i)
                bbox=det_original_in_img_i(det_i).bbox;
                bbox_x1=bbox(1);bbox_y1=bbox(2);bbox_x2=bbox_x1+bbox(3)-1;bbox_y2=bbox_y1+bbox(4)-1;
                category_id=det_original_in_img_i(det_i).category_id;
                %keep the detection of bus (catogery 3)
%                 if category_id==3
%                     det_i_keep=[det_i_keep;det_i];
%                 end
                if bbox_x1>=crop_x1+dis_to_boundary&&bbox_y1>=crop_y1+dis_to_boundary&&bbox_x2<=crop_x2-dis_to_boundary&&bbox_y2<=crop_y2-dis_to_boundary
                else
                    det_i_keep=[det_i_keep;det_i];
                end
            end
            det_original_in_img_i=det_original_in_img_i(det_i_keep);
        end
    end
    % make sure the dimension is consistent
    [d1,~]=size(det_original_in_img);
    [d1_1,~]=size(det_original_in_img_i);
    if d1~=d1_1 && d1_1~=0&&d1~=0
        det_original_in_img_i=det_original_in_img_i';
    end
    det_original_in_img=[det_original_in_img,det_original_in_img_i];
    %show detection from crop regions in original image
    if 0
        img_file=fullfile(img_coco_path,img_list(i).name);
        imshow(img_file);
        hold on
        for det_i=1:length(det_in_img_i)
            rect=det_in_img_i(det_i).bbox(:);
            rectangle('Position',rect,'EdgeColor','r');
        end
        pause(0.05)
        hold off
    end
    %show detection not in crop regions in original image
    if 0
        img_file=fullfile(img_coco_path,img_list(i).name);
        imshow(img_file);
        hold on
        for det_i=1:length(det_original_in_img_i)
            rect=det_original_in_img_i(det_i).bbox(:);
            rectangle('Position',rect,'EdgeColor','b');
        end
        pause(0.05)
        hold off
    end
end
end
%merge detection in clusters and original images
det_in_img=[det_in_img,det_original_in_img];
%apply nms in class-wise way
if (1)
image_id_list=[det_in_img.image_id];
img_ids=unique(image_id_list);
category_id_list=[det_in_img.category_id];
category_ids=unique(category_id_list);
score_list=[det_in_img.score];
bbox_list=[];
det_in_img_nms=[];
for i=1:length(det_in_img)
    bbox_list=[bbox_list;det_in_img(i).bbox];
end
overlap=0.7;
for img_id=img_ids
    img_id;
    %class-wise nms
    if 1
    for cat_id=category_ids
        box_id_in_img_i=find(image_id_list==img_id&category_id_list==cat_id);
        box_score_in_img_i=score_list(box_id_in_img_i);
        bbox_list_in_img_i=bbox_list(box_id_in_img_i,:);
        [box_score_img_i_sort,box_score_img_i_back]=sort(box_score_in_img_i,'descend');
        bbox_list_in_img_i_sorted=bbox_list_in_img_i(box_score_img_i_back,:);
        %[x1,y1,w,h] to [x1,y1,x2,y2]
        x1=bbox_list_in_img_i_sorted(:,1);
        y1=bbox_list_in_img_i_sorted(:,2);
        w=bbox_list_in_img_i_sorted(:,3);
        h=bbox_list_in_img_i_sorted(:,4);
        x2=x1+w-1;
        y2=y1+h-1;
        bbox_list_in_img_i_sorted=[x1,y1,x2,y2];
        boxes=bbox_list_in_img_i_sorted;
        pick = nms(boxes, overlap);
        
        pick=box_id_in_img_i(box_score_img_i_back(pick));
        det_in_img_nms=[det_in_img_nms,det_in_img(pick)];
    end
    end
    %non-class wise nms
    if 0
        box_id_in_img_i=find(image_id_list==img_id);
        box_score_in_img_i=score_list(box_id_in_img_i);
        bbox_list_in_img_i=bbox_list(box_id_in_img_i,:);
        [box_score_img_i_sort,box_score_img_i_back]=sort(box_score_in_img_i,'descend');
        bbox_list_in_img_i_sorted=bbox_list_in_img_i(box_score_img_i_back,:);
        %[x1,y1,w,h] to [x1,y1,x2,y2]
        x1=bbox_list_in_img_i_sorted(:,1);
        y1=bbox_list_in_img_i_sorted(:,2);
        w=bbox_list_in_img_i_sorted(:,3);
        h=bbox_list_in_img_i_sorted(:,4);
        x2=x1+w-1;
        y2=y1+h-1;
        bbox_list_in_img_i_sorted=[x1,y1,x2,y2];
        boxes=bbox_list_in_img_i_sorted;
        pick = nms(boxes, overlap);
        
        pick=box_id_in_img_i(box_score_img_i_back(pick));
        det_in_img_nms=[det_in_img_nms,det_in_img(pick)];
    end
    
end

det_in_img=det_in_img_nms;
end
% fuse detection in crop and out of crop regions
% det_in_img=[det_in_img,det_original_in_img];
% get top N detections in each image
if 1
det_in_img_topN=[];    
N=500;    
score_list=[det_in_img.score];
image_id_list=[det_in_img.image_id];
img_ids=unique(image_id_list);
%select the top score detection
if 0
    score_list=[det_in_img.score];
    idx=find(score_list>0.9);
    score_list=score_list(idx);
    image_id_list=image_id_list(idx);
    img_ids=unique(image_id_list);
end
for img_id=img_ids
    box_id_in_img_i=find(image_id_list==img_id);
    score_list_in_img_i=score_list(box_id_in_img_i);
    [score_list_in_img_i_sorted,score_list_in_img_i_back]=sort(score_list_in_img_i,'descend');
    if length(score_list_in_img_i_back)>N
        score_list_in_img_i_back=score_list_in_img_i_back(1:N);
    end
    pick=box_id_in_img_i(score_list_in_img_i_back);
    det_in_img_topN=[det_in_img_topN,det_in_img(pick)];
end
det_in_img=det_in_img_topN;
end

f=fopen(output_file,'w'); fwrite(f,gason(det_in_img)); fclose(f);
end
