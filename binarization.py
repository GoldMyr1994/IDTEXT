import os
import numpy as np
import cv2

PATH_DATASET = "examples/images/"
PATH_RESULT = "examples/binarization/"
PATH_CONF = "examples/conf"

IMG_NAME_PREFIX = "IMG"
IMG_EXT = "JPG"

def main():

    images = [f for f in os.listdir(PATH_DATASET) if os.path.isfile(os.path.join(PATH_DATASET, f))]
    images = [img for img in images if img.startswith(IMG_NAME_PREFIX)]
    images = [img for img in images if img.endswith(".{}".format(IMG_EXT))]

    for e in images:
        img_name = os.path.splitext(e)[0]
        print(e,img_name)
        img = cv2.imread("{}{}".format(PATH_DATASET,e),0)
        _,ret = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        cv2.imwrite("{}{}.{}".format(PATH_RESULT,img_name,".jpg"),ret)

if __name__ == "__main__":
    main()





