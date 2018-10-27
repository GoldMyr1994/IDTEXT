import os
from os import listdir
from os.path import isfile, join
import numpy as np


class Box:
    def __init__(self,x,y,w,h,angle):
        self.start_x = x
        self.start_y = y
        self.w = w
        self.h = h
        self.stop_x = self.start_x+self.w
        self.stop_y = self.start_y+self.h
        self.angle = angle
        self.area = self.w*self.h


    @staticmethod
    def hasIntersection(box1,box2):
        if box1.start_x > box2.stop_x:
            return False
        if box2.start_x > box1.stop_x:
            return False

        if box1.start_y > box2.stop_y:
            return False
        if box2.start_y > box1.stop_y:
            return False

        return True

    @staticmethod
    def intersection(box1,box2):
        start_x = max(box1.start_x,box2.start_x)
        start_y = max(box1.start_y,box2.start_y)
        stop_x = min(box1.stop_x,box2.stop_x)
        stop_y = min(box1.stop_y,box2.stop_y)
        w = stop_x-start_x
        h = stop_y-start_y
        return Box(start_x,start_y,w,h,0)


PATH_DATASET = "examples/images/"
PATH_RESULT = "examples/results"
PATH_CONF = "examples/conf"

IMG_NAME_PREFIX = "IMG"
IMG_EXT = "JPG"


def measure(img_name):

    n_positive = 0.
    n_negative = 0.

    n_truth = 0.
    n_estimated = 0.


    img_name = os.path.splitext(img_name)[0]
    print("====================== ", img_name, " ======================")

    gt_path = join(PATH_DATASET,"{}.{}".format(img_name,"gt"))
    gt_s, gt_list = "", []

    with open(gt_path) as f: 
        gt_s = f.readlines()

    for line in gt_s:
        label, difficulty, x, y, w, h, angle = line.split(" ")
        label, difficulty, x, y, w, h, angle = int(label), int(difficulty), int(x), int(y), int(w), int(h), float(angle)
        gt_list.append( (Box(x,y,w,h,angle), label, difficulty, x, y, w, h, angle) )

    g_path = join(PATH_RESULT,img_name,"{}.{}".format("gt","gt"))
    if not os.path.isfile(g_path): 
            return 0., 0., 0., 0., 0., 0., 0.
    g_s, g_list = "", []

    with open(g_path) as f: 
        g_s = f.readlines()

    for line in g_s:
        label, difficulty, x, y, w, h, angle, _ = line.split(" ")
        label, difficulty, x, y, w, h, angle = int(label), int(difficulty), int(x), int(y), int(w), int(h), float(angle)
        g_list.append( (Box(x,y,w,h,angle), label, difficulty, x, y, w, h, angle) )
    

    n_truth += len(gt_list) 
    n_estimated += len(g_list) 

    for box_gt_data in gt_list: 
        box_gt = box_gt_data[0]

        positive = False
        for box_g_data in g_list:
            box_g = box_g_data[0]

            if Box.hasIntersection(box_g,box_gt):
                if abs(box_g.angle-box_gt.angle)<np.pi/8:
                    box_int = Box.intersection(box_g,box_gt)
                    ratio = float(box_int.area)/( float(box_g.area) + float(box_gt.area) - float(box_int.area))
                    if ratio > 0.5:
                        n_positive += 1
                        positive = True
                        break

        if not positive:
            if box_gt_data[2] == 1:
                n_truth -= 1
            #n_negative += 1

    if n_positive == 0:
        return 0., 0., 0., 0., 0., 0., 0.

    precision = n_positive/n_estimated
    recall = n_positive/n_truth
    Fmeasure = 2*precision*recall/(precision+recall)
    print("precisione: {}, recall: {}, F-measure: {}".format(precision, recall, Fmeasure))
    if precision<0.2 or recall<0.1:
        print("bad result", img_name)
        print()
        return 0., 0., 0., 0., 0., 0., 0.
    print()
    return n_positive, n_negative, n_truth, n_estimated, precision, recall, Fmeasure


def main():

    images = [f for f in os.listdir(PATH_DATASET) if os.path.isfile(os.path.join(PATH_DATASET, f))]
    images = [img for img in images if img.startswith(IMG_NAME_PREFIX)]
    images = [img for img in images if img.endswith(".{}".format(IMG_EXT))]

    n_positive = 0.
    n_negative = 0.

    n_truth = 0.
    n_estimated = 0.

    total = 0

    for img in images:
        np, nn, nt, ne, _, _, _ = measure(img)

        n_positive += np
        n_negative += nn
        n_truth += nt
        n_estimated += ne

        if np >= 1: 
            total += 1

    precision = n_positive/n_estimated
    recall = n_positive/n_truth
    Fmeasure = 2*precision*recall/(precision+recall)

    print("#img : ", total)
    print("precisione: {}, recall: {}, F-measure: {}".format(precision, recall, Fmeasure))


if __name__ == "__main__":
    main()





