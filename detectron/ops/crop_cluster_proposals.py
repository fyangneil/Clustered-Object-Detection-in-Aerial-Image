"""
crop the cluster proposals
-------------------------------
Written by Fan Yang
-------------------------------
"""
import pickle,json,os,cv2
import numpy as np,glob
from ClusterMerge import ICM,NMM
def load_cluster_proposals(proposal_file):
    """
    :param proposal_file: cluster proposal_file:
    :return: cluster proposals
    """
    with open(proposal_file, 'rb') as f:
        data = pickle.load(f)
        print('load the cluster proposals done')
        return data
def load_cluster_detections(cluster_detection_file):
    """
    :param proposal_file: cluster proposal_file:
    :return: cluster proposals
    """
    with open(cluster_detection_file, 'rb') as f:
        data = pickle.load(f)
        print('load the cluster detections done')
        return data
def load_cluster_detections_json(cluster_detection_file):
    """
    :param gt_file: ground truth file:
    :return: ground truth data
    """
    cluster_det=open(cluster_detection_file,'r')
    data=json.load(cluster_det)
    print('load the cluster detections done')
    return data

def load_gt(gt_file):
    """
    :param gt_file: ground truth file:
    :return: ground truth data
    """
    gt_file=open(gt_file,'r')
    data=json.load(gt_file)
    print('load ground truth done')
    return data
def show_proposal(gt,proposal,img_path):
    """
    :param gt: ground truth data
    :param proposal: cluster proposals data
    :return:
    """

    img_ids=proposal['ids']
    proposal_bboxes=proposal['boxes']
    proposal_scores = proposal['scores']
    img_inf=gt['images']
    for i in range(len(img_ids)):
        if img_ids[i]==img_inf[i]['id']:
            img_name=img_inf[i]['file_name']
            img_file=os.path.join(img_path,img_name)
            img = cv2.imread(img_file)
            # draw bbox on image
            proposal_bboxes_in_img_i=proposal_bboxes[i]
            proposal_bboxes_in_img_i=proposal_bboxes_in_img_i[proposal_scores[i]>0.1]
            proposal_bboxes_in_img_i = ICM(proposal_bboxes_in_img_i, 0.7, 1)
            for bbox_i, bbox in enumerate(proposal_bboxes_in_img_i):
                bbox_x1 = int(bbox[0])
                bbox_y1 = int(bbox[1])
                bbox_x2 = int(bbox[2])
                bbox_y2 = int(bbox[3])
                cv2.rectangle(img, (bbox_x1, bbox_y1), (bbox_x2, bbox_y2), (0, 255, 0), 2)
            cv2.imshow('image', img)

            cv2.waitKey(0)
            cv2.destroyAllWindows()
def show_detection(gt,detection,img_path):
    """
    :param gt: ground truth data
    :param proposal: cluster proposals data
    :return:
    """
    det_bboxes_scores_all=detection['all_boxes'][1]

    img_inf=gt['images']
    for i in range(len(det_bboxes_scores_all)):
        det_bboxes_scores=det_bboxes_scores_all[i][:,:5]
        det_scores = det_bboxes_scores_all[i][:,4]
        if len(det_scores)>0:
            img_name=img_inf[i]['file_name']
            img_file=os.path.join(img_path,img_name)
            img = cv2.imread(img_file)
            # draw bbox on image
            det_bboxes_scores=det_bboxes_scores[det_scores>0.3,:]
            det_bboxes_scores = ICM(det_bboxes_scores, 0.5, 3)
            # det_bboxes_scores = NMM(det_bboxes_scores, 0.7)
            # det_bboxes = NMM(det_bboxes, 0.7)

            for bbox_i, bbox in enumerate(det_bboxes_scores):
                bbox_x1 = int(bbox[0])
                bbox_y1 = int(bbox[1])
                bbox_x2 = int(bbox[2])
                bbox_y2 = int(bbox[3])
                cv2.rectangle(img, (bbox_x1, bbox_y1), (bbox_x2, bbox_y2), (0, 255, 0), 2)
            cv2.imshow('image', img)

            cv2.waitKey(0)
            cv2.destroyAllWindows()


def NMS(boxes,iouthreshold):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = range(len(area))#np.argsort(y2)
    idxs.reverse()
    # keep looping while some indexes still remain in the indexes
    # list

    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / (area[idxs[:last]]+area[last]-w*h)


        # delete all indexes from the index list that have
        delete_idxs = np.concatenate(([last],
                                      np.where(overlap > iouthreshold)[0]))
        idxs = np.delete(idxs, delete_idxs)


    # return only the bounding boxes that were picked using the
    # integer data type

    return boxes[pick].astype("int")
def crop_cluster_meanshift(gt_file_list,img_path,gt_save_path,img_save_path):
    """
    crop the clustered regions generated by meanshift
    """
    if not os.path.exists(gt_save_path):
        os.mkdir(gt_save_path)
    if not os.path.exists(img_save_path):
        os.mkdir(img_save_path)


    for i,gt_file in enumerate(gt_file_list):
        print(gt_file)

        gt=np.loadtxt(gt_file, delimiter=',')
        gt=gt.astype(np.int)
        if len(gt.shape)==1:
            continue
        object_gt_inds=np.where((gt[:,5]<11) & (gt[:,4]==1))[0]
        object_gt=gt[object_gt_inds,:]
        cluster_gt_inds = np.where(gt[:,5] == 11)[0]
        cluster_gt=gt[cluster_gt_inds,:]

        # get cluster bounding box [x1,y1,x2,y2]
        img_name=os.path.splitext(os.path.basename(gt_file))[0]
        img_file=os.path.join(img_path, img_name+'.jpg')
        img = cv2.imread(img_file)
        cluster_x1=cluster_gt[:,0]
        cluster_y1 =cluster_gt[:,1]
        cluster_x2 =cluster_gt[:,0]+cluster_gt[:,2]-1
        cluster_y2 =cluster_gt[:,1]+cluster_gt[:,3]-1
        cluster_num=len(cluster_gt[:,0])
        if 1:
            # show cluster on raw image
            for cluster_i in range(cluster_num):
                cv2.rectangle(img, (cluster_x1[cluster_i], cluster_y1[cluster_i]), (cluster_x2[cluster_i], cluster_y2[cluster_i]), (0, 255, 0), 2)
            cv2.imshow('image', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        # get the center point of the objects
        obj_x1=object_gt[:,0]
        obj_y1 = object_gt[:, 1]
        obj_w = object_gt[:, 2]
        obj_h = object_gt[:, 3]
        obj_x2 = obj_x1+obj_w-1
        obj_y2 = obj_y1+obj_h-1
        obj_center_x=obj_x1+obj_w/2-1
        obj_center_x=obj_center_x.astype(np.int)
        obj_center_y = obj_y1 + obj_h/2-1
        obj_center_y=obj_center_y.astype(np.int)

        obj_score=object_gt[:,4]
        obj_cls = object_gt[:, 5]
        obj_truncation=object_gt[:,6]
        obj_occlusion = object_gt[:, 7]
        # crop cluster regions
        for cluster_i in range(cluster_num):
            cluster_i_x1 = cluster_x1[cluster_i]
            cluster_i_y1 = cluster_y1[cluster_i]
            cluster_i_x2 = cluster_x2[cluster_i]
            cluster_i_y2 = cluster_y2[cluster_i]
            cluster_i_w=cluster_i_x2-cluster_i_x1+1
            cluster_i_h=cluster_i_y2-cluster_i_y1+1

            obj_ids_in_cluster_i=np.where((obj_center_x<cluster_i_x2) & (obj_center_x>cluster_i_x1)
                     & (obj_center_y<cluster_i_y2) & (obj_center_y>cluster_i_y1))[0]
            obj_num_in_cluster_i=len(obj_ids_in_cluster_i)
            if obj_num_in_cluster_i>3:

                img_cluster_region=img[cluster_y1[cluster_i]:cluster_y2[cluster_i], cluster_x1[cluster_i]:cluster_x2[cluster_i]]

                # show cropped cluster region
                if 1:
                    cv2.imshow('image_cluster_region', img_cluster_region)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                # get the object location [x1,y1,w,h] in cluster region
                obj_x1_cluster_i = np.maximum(obj_x1[obj_ids_in_cluster_i] - cluster_i_x1,0)
                obj_y1_cluster_i = np.maximum(obj_y1[obj_ids_in_cluster_i] - cluster_i_y1,0)
                obj_x2_cluster_i = np.minimum(obj_x2[obj_ids_in_cluster_i] - cluster_i_x1,cluster_i_w)
                obj_y2_cluster_i = np.minimum(obj_y2[obj_ids_in_cluster_i] - cluster_i_y1,cluster_i_h)
                obj_w_cluster_i=obj_x2_cluster_i-obj_x1_cluster_i
                obj_h_cluster_i = obj_y2_cluster_i - obj_y1_cluster_i

                # show object in the cropped region
                if 1:
                    for obj_i_in_cluster_i in range(obj_num_in_cluster_i):
                        cv2.rectangle(img_cluster_region, (obj_x1_cluster_i[obj_i_in_cluster_i], obj_y1_cluster_i[obj_i_in_cluster_i]),
                                      (obj_x2_cluster_i[obj_i_in_cluster_i], obj_y2_cluster_i[obj_i_in_cluster_i]), (0, 255, 0), 2)
                    cv2.imshow('image_cluster_region', img_cluster_region)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                # save object information to text file
                object_gt_in_cluster_i=object_gt[obj_ids_in_cluster_i,:]
                object_gt_in_cluster_i[:,0]=obj_x1_cluster_i
                object_gt_in_cluster_i[:, 1] = obj_y1_cluster_i
                object_gt_in_cluster_i[:, 2] = obj_w_cluster_i
                object_gt_in_cluster_i[:, 3] = obj_h_cluster_i
                obj_inf=object_gt_in_cluster_i
                cluster_gt_file=os.path.join(gt_save_path,img_name+'_{}.txt'.format(cluster_i))

                np.savetxt(cluster_gt_file, obj_inf, fmt='%u',delimiter=',')
                # save cropped cluster image
                cluster_img_file=os.path.join(img_save_path,img_name+'_{}_meanshift.jpg'.format(cluster_i))
                cv2.imwrite(cluster_img_file,img_cluster_region)

def crop_cluster_ClusterNet(cluster_det,gt,img_path,img_save_path,gt_save_path):
    """
    crop the clustered regions generated by ClusterNet
    """
    if not os.path.exists(gt_save_path):
        os.mkdir(gt_save_path)
    if not os.path.exists(img_save_path):
        os.mkdir(img_save_path)

    det_bboxes_scores_all = cluster_det['all_boxes'][1]

    img_inf = gt['images']
    for i in range(len(det_bboxes_scores_all)):
        det_bboxes_scores = det_bboxes_scores_all[i][:, :5]
        det_scores = det_bboxes_scores_all[i][:, 4]
        if len(det_scores) > 0:
            img_name = img_inf[i]['file_name']
            img_file = os.path.join(img_path, img_name)
            img = cv2.imread(img_file)
            det_bboxes_scores = det_bboxes_scores[det_scores > 0.3, :]
            det_bboxes_scores = ICM(det_bboxes_scores, 0.7, 3)

            # draw bbox on image
            if 0:
                for bbox_i, bbox in enumerate(det_bboxes_scores):
                    bbox_x1 = int(bbox[0])
                    bbox_y1 = int(bbox[1])
                    bbox_x2 = int(bbox[2])
                    bbox_y2 = int(bbox[3])
                    cv2.rectangle(img, (bbox_x1, bbox_y1), (bbox_x2, bbox_y2), (0, 255, 0), 2)
                cv2.imshow('image', img)

                cv2.waitKey(0)
                cv2.destroyAllWindows()
            # crop cluster regions
            for bbox_i, bbox in enumerate(det_bboxes_scores):
                bbox_x1 = int(bbox[0])
                bbox_y1 = int(bbox[1])
                bbox_x2 = int(bbox[2])
                bbox_y2 = int(bbox[3])
                img_cluster_region = img[bbox_y1:bbox_y2,
                                 bbox_x1:bbox_x2]
                # show cropped cluster region
                if 0:
                    cv2.imshow('image_cluster_region', img_cluster_region)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                # save cropped cluster image
                img_name=os.path.splitext(os.path.basename(img_name))[0]

                cluster_img_file = os.path.join(img_save_path, img_name + '_{}_{}.jpg'.format(bbox_x1,bbox_y1))
                cv2.imwrite(cluster_img_file, img_cluster_region)
                # save object information to text file
                obj_inf=np.array([[1,1,10,10,1,1,1,1],[1,1,10,10,1,1,1,1]]) # the valute do not have practical meanning
                cluster_gt_file=os.path.join(gt_save_path,img_name+'_{}_{}.txt'.format(bbox_x1,bbox_y1))

                np.savetxt(cluster_gt_file, obj_inf, fmt='%u',delimiter=',')


if __name__ == "__main__":
    cluster_proposal_file='/media/fanyang/C/code/detector/Detectron-Cascade-RCNN/test/visdrone/coco_2014_val' \
                          '/generalized_rcnn/detections.pkl'
    gt_file='/media/fanyang/C/data/visdrone/VisDrone2018-DET-val/coco_format_annotations/instances_val2014.json'
    img_path='/media/fanyang/C/data/visdrone/VisDrone2018-DET-val/images_coco_format_rename'
    gt_save_path='/media/fanyang/C/data/visdrone/VisDrone2018-DET-val/cluster_crop_gt_CPNet'
    img_save_path='/media/fanyang/C/data/visdrone/VisDrone2018-DET-val/cluster_crop_image_CPNet'
    # cluster_proposal=load_cluster_proposals(cluster_proposal_file)
    cluster_detection=load_cluster_detections(cluster_proposal_file)
    gt=load_gt(gt_file)
    crop_cluster_ClusterNet(cluster_detection,gt,img_path,img_save_path,gt_save_path)

