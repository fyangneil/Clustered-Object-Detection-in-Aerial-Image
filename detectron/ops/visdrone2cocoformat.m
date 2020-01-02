function visdrone2cocoformat
%% convert ground truth from visdrone to coco format
addpath('./cocoapi-master/MatlabAPI');
addpath('./meanshift');

cls_name={'pedestrian', 'people', 'bicycle', 'car', 'van','truck', 'tricycle', 'awning-tricycle', 'bus', 'motor','cluster'};

gt_path='/media/fanyang/C/data/visdrone/VisDrone2018-DET-val/cluster_crop_gt_CPNet';
gt_list=dir(fullfile(gt_path,'*txt'));
im_path='/media/fanyang/C/data/visdrone/VisDrone2018-DET-val/cluster_crop_image_CPNet';
output_path='/media/fanyang/C/data/visdrone/VisDrone2018-DET-val/coco_format_annotations';
output_file=fullfile(output_path,'cluster_CPNet.json');
cls_num=length(cls_name);
if ~exist(output_path)
    mkdir(output_path)
end
img_list=dir(fullfile(im_path,'*jpg'));
img_num=length(img_list);
obj_num=0;
clk=tic;
id_count=1;
im_name_list=[];
id_list=[],hw_list=[],catIds_list=[],ignore_list=[],iscrowd_list=[],bbs_list=[];
test_flag=0;
superclass=0;
if test_flag
gt_list=img_list;
end
gt_num=length(gt_list);
data=CocoUtils.initData(cls_name,gt_num);

for i=1:gt_num 

    im_name=[gt_list(i).name(1:end-4) '.jpg'];
    i
    im=imread(fullfile(im_path,im_name));
    [h,w,c]=size(im);

    hw=[h,w];
    bbs=[];
    catIds=[];
    ignore=[];
    iscrowd=[];
    valid_num=0;
    if test_flag==1
       id=i;
       bbs=[1,1,w-1,h-1];%% x_min and y_min should be set to 1 instead of 0
       catIds=1;
       ignore=1;
       iscrowd=1;
       data=CocoUtils.addData(data,im_name,id,hw,catIds,ignore,iscrowd,bbs);
       continue;
    end
    %id=is{i}; id(id=='_')=[]; id=str2double(id);
    if isempty(gt_list(i).bytes) || gt_list(i).bytes==0 
       id=i;
       bbs=[1,1,w-1,h-1];%% x_min and y_min should be set to 1 instead of 0
       catIds=1;
       ignore=1;
       iscrowd=1;
       data=CocoUtils.addData(data,im_name,id,hw,catIds,ignore,iscrowd,bbs);
       continue 
    end
     gts=importdata(fullfile(gt_path,gt_list(i).name));
    if isempty(gts)
        
       id=i;
       bbs=[1,1,w-1,h-1];%% x_min and y_min should be set to 1 instead of 0
       catIds=1;
       ignore=1;
       iscrowd=1;
       data=CocoUtils.addData(data,im_name,id,hw,catIds,ignore,iscrowd,bbs); 
       
    else
     % reassign gt label
        if superclass==2
            gts(gts(:,6)~=1&gts(:,6)~=2,6)=0;
            gts(gts(:,6)~=1&gts(:,6)~=2,5)=0;
        end
        if superclass==8
            gts(gts(:,6)==2,6)=1;
            gts(gts(:,6)==3,6)=2;
            gts(gts(:,6)==4,6)=3;
            gts(gts(:,6)==5,6)=4;
            gts(gts(:,6)==6,6)=5;
            gts(gts(:,6)==7,6)=6;
            gts(gts(:,6)==8,6)=7;
            gts(gts(:,6)==9,6)=8;
            gts(gts(:,6)==10,6)=9;
        end
        % car(4), van(5), truck(6), bus(9)
        if superclass==4
            %set score to zero
           gts(gts(:,6)~=4&gts(:,6)~=5&gts(:,6)~=6&gts(:,6)~=9,5)=0;
           gts(gts(:,6)==4,6)=1;
           gts(gts(:,6)==5,6)=2;
           gts(gts(:,6)==6,6)=3;
           gts(gts(:,6)==9,6)=4;
        end
    for gt_ind=1:length(gts(:,1))
        bbox_left=gts(gt_ind,1);
        bbox_top=gts(gt_ind,2);
        bbox_width=gts(gt_ind,3);
        bbox_height=gts(gt_ind,4);
        bbox_right=bbox_left+bbox_width-1;
        bbox_bottom=bbox_top+bbox_height-1;
        if bbox_left<=0
%         disp('bbox_left exceed the image boundary');
        bbox_left=1;
        end
        if bbox_top<=0
%         disp('bbox_top exceed the image boundary');
        bbox_top=1;
        end
        
        if bbox_right>w
%         disp('bbox_right exceed the image boundary');
        bbox_right=w;
        end
        
        if bbox_bottom>h
%         disp('bbox_bottom exceed the image boundary');
        bbox_bottom=h;
        end
        
        score=gts(gt_ind,5);
        
        category=gts(gt_ind,6);
       

       %if score
       %  valid_num=valid_num+1; 
        bbs(gt_ind,:)=[bbox_left,bbox_top,bbox_right,bbox_bottom]; 
        
        if(score)
            %% valid region
            catIds(gt_ind)=category;
            iscrowd(gt_ind)=0;
        else
            %% not valid region
            catIds(gt_ind)=1;
            iscrowd(gt_ind)=1;
        end
        ignore(gt_ind)=iscrowd(gt_ind);
       %end
     end
     id=i;
     data=CocoUtils.addData(data,im_name,id,hw,catIds,ignore,iscrowd,bbs);
     

    end
end
f=fopen(output_file,'w'); fwrite(f,gason(data)); fclose(f);
fprintf('DONE (t=%0.2fs).\n',toc(clk));
end
