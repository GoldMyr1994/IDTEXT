import os
import json


PATH_DATASET = "examples/images/"
PATH_RESULT = "examples/results"
PATH_CONF = "examples/conf"

IMG_NAME_PREFIX = "IMG"
IMG_EXT = "JPG"

images = []

images = [f for f in os.listdir(PATH_DATASET) if os.path.isfile(os.path.join(PATH_DATASET, f))]
images = [img for img in images if img.startswith(IMG_NAME_PREFIX)]
images = [img for img in images if img.endswith(".{}".format(IMG_EXT))]

run_txt_dol = ""
run_txt_lod = ""

for img in images:
    
    img_name = os.path.splitext(img)[0]
    img_name_number = img_name.split('_')[1]

    print("====================== ", img_name, " ======================")
    
    cfg_name = '{}/configim_{}.json'.format(PATH_CONF,img_name.split('_')[1])
    cfg = None

    with open(cfg_name) as f:
        cfg = json.load(f)
    
    if cfg["dark_on_light"]:
        run_txt_dol += "python idtext.py {} \n".format(cfg_name)
    else:
        run_txt_lod += "python idtext.py {} \n".format(cfg_name)

with open("run_dol.bat", 'w') as fp:
    fp.write(run_txt_dol)

with open("run_lod.bat", 'w') as fp:
    fp.write(run_txt_lod)