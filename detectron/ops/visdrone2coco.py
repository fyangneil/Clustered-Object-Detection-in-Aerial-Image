import os,glob
import json
import argparse
import PIL.Image
import numpy as np


def parse_arguments():
    parser = argparse.ArgumentParser(description='VisDrone to COCO format')
    parser.add_argument(
          "-g", "--gt_dir",
          default="/media/fanyang/C/data/visdrone/VisDrone2018-DET-train/eip_meanshift_aug_gt",
          help="root directory of BDD label Json files",
    )
    parser.add_argument(
        "-i", "--image_dir",
        default="/media/fanyang/C/data/visdrone/VisDrone2018-DET-train/eip_meanshift_aug_image",
        help="root directory of BDD label Json files",
    )
    parser.add_argument(
          "-s", "--save_path",
          default="/media/fanyang/C/data/visdrone/VisDrone2018-DET-train/coco_format_annotations",
          help="path to save coco formatted label file",
    )
    return parser.parse_args()



def visdrone2coco_detection(image_file_list,gt_file_list, fn):

    images = list()
    annotations = list()

    counter = 0
    gt_counter=0
    for i,entry in enumerate(image_file_list):
        print(i)
        img = PIL.Image.open(entry)
        counter += 1
        image = dict()
        file_name=os.path.basename(entry)
        # file_name=os.path.split(file_name)[0]
        image['file_name'] = file_name
        image['height'] = img.size[1]
        image['width'] = img.size[0]

        image['id'] = counter

        if os.stat(gt_file_list[i]).st_size == 0:
            gt_inform=np.array([1,1,10,10,0,0,0,0])
        else:
            gt_inform=np.loadtxt(gt_file_list[i],delimiter=',')

        if len(gt_inform.shape)==1:
            gt_inform=gt_inform.reshape((1,gt_inform.shape[0]))

        for gt_i in range(gt_inform.shape[0]):
            gt_counter +=1
            annotation = dict()
            x1=gt_inform[gt_i,0]
            y1 = gt_inform[gt_i, 1]
            w = gt_inform[gt_i, 2]
            h = gt_inform[gt_i, 3]
            x2=x1+w-1
            y2=y1+h-1
            score=gt_inform[gt_i, 4]
            cat=gt_inform[gt_i, 5]

            if score==0 or cat==0 or cat==11:
                annotation["iscrowd"] = 1
            else:
                annotation["iscrowd"] = 0


            annotation["image_id"] = i+1
            annotation['bbox'] = [x1, y1, w, h]
            annotation['area'] = float((x2 - x1) * (y2 - y1))
            annotation['category_id'] = cat
            annotation['ignore'] = annotation["iscrowd"]
            annotation['id'] =gt_counter
            annotation['segmentation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
            annotations.append(annotation)


        images.append(image)

    attr_dict["images"] = images
    attr_dict["annotations"] = annotations
    attr_dict["type"] = "instances"

    print('saving...')
    json_string = json.dumps(attr_dict)
    with open(fn, "w") as file:
        file.write(json_string)


if __name__ == '__main__':

    args = parse_arguments()

    attr_dict = dict()
    attr_dict["categories"] = [
        {"supercategory": "none", "id": 1, "name": "pedestrian"},
        {"supercategory": "none", "id": 2, "name": "people"},
        {"supercategory": "none", "id": 3, "name": "bicycle"},
        {"supercategory": "none", "id": 4, "name": "car"},
        {"supercategory": "none", "id": 5, "name": "van"},
        {"supercategory": "none", "id": 6, "name": "truck"},
        {"supercategory": "none", "id": 7, "name": "tricycle"},
        {"supercategory": "none", "id": 8, "name": "awning - tricycle"},
        {"supercategory": "none", "id": 9, "name": "bus"},
        {"supercategory": "none", "id": 10, "name": "motor"}
    ]


    # create visdrone annotation file in COCO format
    print('Loading ground truth file...')
    gt_file_path=os.path.join(args.gt_dir,'*.txt')
    gt_file_list=glob.glob(gt_file_path)
    gt_file_list.sort()
    print('Loading image file...')
    image_file_path = os.path.join(args.image_dir, '*.jpg')
    image_file_list = glob.glob(image_file_path)
    image_file_list.sort()

    out_fn = os.path.join(args.save_path,
                          'eip_meanshift_aug_py.json')
    visdrone2coco_detection(image_file_list,gt_file_list, out_fn)

