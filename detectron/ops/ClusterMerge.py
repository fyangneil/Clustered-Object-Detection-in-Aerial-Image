import numpy as np
# from visualize_cluster_proposals import load_cluster_detections,load_gt,show_detection


def ICM(boxes,iouthreshold,topN):
    """
    """
    boxes_num=len(boxes[:,0])
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    while boxes_num>topN:

        boxes=NMM(boxes, iouthreshold)
        boxes_num_after_merge=len(boxes[:,0])
        if boxes_num==boxes_num_after_merge:
            boxes_num_min=np.minimum(boxes_num_after_merge,topN)
            boxes=boxes[:boxes_num_min]
            break
        else:
            boxes_num=boxes_num_after_merge
    return boxes

def NMM(boxes,iouthreshold):
    """
    """
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # sort the boxes according to score
    scores=boxes[:,4]
    idxs=np.argsort(-scores)
    boxes=boxes[idxs,:]
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
    idxs = range(len(area))  # np.argsort(y2)
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
        overlap = (w * h) / area[idxs[:last]]

        # merge all boxes with overlap larger than threshold
        delete_idxs = np.concatenate(([last],
                                       np.where(overlap > iouthreshold)[0]))
        idxs=np.array(idxs)
        xx1 = np.amin(x1[idxs[delete_idxs]])
        yy1 = np.amin(y1[idxs[delete_idxs]])
        xx2 = np.amax(x2[idxs[delete_idxs]])
        yy2 = np.amax(y2[idxs[delete_idxs]])

        boxes[i,:4]=np.array([xx1,yy1,xx2,yy2])

        # delete all indexes from the index list that have
        idxs=np.delete(idxs,delete_idxs)

    # return only the bounding boxes that were picked using the
    # integer data type

    return boxes[pick].astype("int")
if __name__ == "__main__":


    cluster_proposal_file = '/media/fanyang/C/code/detector/Detectron-Cascade-RCNN' \
                            '/test/coco_2014_val/generalized_rcnn/detections.pkl'
    gt_file = '/media/fanyang/C/code/detector/' \
              'Detectron-Cascade-RCNN/detectron/datasets/' \
              'data/coco/annotations/instances_val2014.json'
    img_path = '/media/fanyang/C/code/detector/' \
               'Detectron-Cascade-RCNN/detectron/datasets/data/coco/coco_val2014'
    cluster_detection = load_cluster_detections(cluster_proposal_file)
    gt = load_gt(gt_file)
    show_detection(gt,cluster_detection,img_path)
