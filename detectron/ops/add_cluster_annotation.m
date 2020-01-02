function add_cluster_annotation
%% add cluster ground truth to original annotation
%% cluster ground truth is labeled as 11
addpath('./MeanShift');
gt_txt_path=['/media/fanyang/C/data/visdrone/VisDrone2018-DET-train/annotations_coco_format_rename'];
img_path='/media/fanyang/C/data/visdrone/VisDrone2018-DET-train/images_coco_format_rename';
gt_output_path='/media/fanyang/C/data/visdrone/VisDrone2018-DET-train/annotations_add_cluster_coco_format_rename';
gt_file_list=dir(fullfile(gt_txt_path,'*txt'));
clus_top_n=10;
dota_flag=0;
only_cluster=0;
bandWidth=0.1;
if ~exist(gt_output_path)
    mkdir(gt_output_path);
end
for ind=1:length(gt_file_list)
    ind
    if isempty(gt_file_list(ind).bytes) || gt_file_list(ind).bytes==0 
       continue 
    end
    %read annotations
    gt_file=fullfile(gt_txt_path,gt_file_list(ind).name);
    gts=dlmread(gt_file);
    gts=gts(gts(:,5)==1,:);
    %get gt of movable object in dota dataset
    %plane(1),ship(2),large-vehicle(10),small-vehicle(11),helicopter(12),
    if dota_flag
       gt_bool=(gts(:,6)==1)|(gts(:,6)==2)|(gts(:,6)==3)|(gts(:,6)==4)|(gts(:,6)==5);
       gts=gts(gt_bool,:); 
    end
    dataPts=gts(:,1:2)+gts(:,3:4)/2-1;
    %read image
    img_file=fullfile(img_path,[gt_file_list(ind).name(1:end-3),'jpg']);
    img=imread(img_file);

    [h,w,~]=size(img);
    dataPts=[dataPts(:,1)/w,dataPts(:,2)/h]';
    
    plotFlag=false;
    [clustCent,data2cluster,~] = MeanShiftCluster(dataPts,bandWidth,plotFlag);
    if isempty(clustCent)
        continue;
    end
    %get box of cluster
    cluster_num=length(clustCent(1,:));
    gt_num_in_cluster=[];
    for i=1:cluster_num
        gt_num_in_cluster(i)=length(find(data2cluster==i));
    end
    [~,socred_ind]=sort(gt_num_in_cluster,'descend');
    cluster_id_list=socred_ind;
    %select the top N largest cluster
    if cluster_num>clus_top_n
        cluster_num=clus_top_n;
    end
    cluster_box_list=[];
    for clus_ind=cluster_id_list(1:cluster_num)
        
        object_gts=gts(data2cluster==clus_ind,1:8);
        %fuse object box to generate cluster box
        if length(object_gts(:,1))<=3
            continue;
        else
            x1=object_gts(:,1);
            y1=object_gts(:,2);
            w=object_gts(:,3);
            h=object_gts(:,4);
            x2=x1+w-1;
            y2=y1+h-1;
            x1=min(x1);
            y1=min(y1);
            x2=max(x2);
            y2=max(y2);
            w=x2-x1+1;
            h=y2-y1+1;
        end
        cluster_box=[x1,y1,w,h];
        score=1;
        if dota_flag
        object_category=1;
        else
         object_category=11;   
        end
        truncation=0;
        occlusion=0;
        cluster_box=[cluster_box,score,object_category,truncation,occlusion];        
        cluster_box_list=[cluster_box_list;cluster_box];
    end
    %add cluster box to gt
    gts=dlmread(gt_file);
    gts=[gts(:,1:8);cluster_box_list];
    if only_cluster
    gts=cluster_box_list;
    end
    % show cluster box
    if 0
    imshow(img);
    hold on;
    for i=1:length(gts(:,1))
        if gts(i,5)==0
            continue;
        end
        if gts(i,6)==16
           rectangle('Position',gts(i,1:4),'EdgeColor','b');
        else
           rectangle('Position',gts(i,1:4),'EdgeColor','r');
        end
    end
     hold off;
    pause(0.1)
    end
%     %save gts with cluster annotations
    output_file=fullfile(gt_output_path,gt_file_list(ind).name);
    dlmwrite(output_file,gts);

end
end
