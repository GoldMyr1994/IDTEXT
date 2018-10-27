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

n_created = 0
n_skipped = 0

for img in images:
    
    img_name = os.path.splitext(img)[0]
    img_name_number = img_name.split('_')[1]

    print("====================== ", img_name, " ======================")
    
    cfg_name = '{}/configim_{}.json'.format(PATH_CONF,img_name.split('_')[1])

    if os.path.isfile(cfg_name): 
        n_skipped += 1
        print(cfg_name, "already exists")
        continue

    n_created += 1
    cfg = {
        "input" : "{}.{}".format(os.path.join(PATH_DATASET,img_name),"JPG"),
        "save": True,
        "output" : "{}".format(os.path.join(PATH_RESULT,img_name)),
        
        "deskew": True,
        "dark_on_light": True,
        
        "letters":{
            "min_width" : 20,
            "min_height" : 20,
            "max_width": 0.4,
            "max_height": 0.4,
            "width_height_ratio" : 5.0,
            "height_width_ratio" : 5.0,
            "min_diag_mswt_ratio" : 3.0,
            "max_diag_mswt_ratio" : 21.0
        },

        "words": {
            "thresh_pairs_y" : 20,
            "thresh_mswt" : 10,
            "thresh_height" : 50,
            "width_scale": 3.0
        },

        "swt_skip_edges": False,
        "gt": True,
    }

    with open(cfg_name, 'w') as fp:
       json.dump(cfg, fp, indent=4)

print("#created: {}\n#skipped: {}".format(n_created,n_skipped))