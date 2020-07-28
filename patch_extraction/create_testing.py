import os
from PIL import Image
import matplotlib.pyplot as plt


testing_case_txt_50 = "/projects/shart/digital_pathology/data/PenMarking/WSIs/testing_cases_50_cv_1.txt" # directory where saves WSI training cases
patch_root_dir = "/projects/shart/digital_pathology/data/PenMarking/WSIs_Img_pairs_256"   # patches root directory need to be restored
test_data = "/projects/shart/digital_pathology/data/PenMarking/testing_data"    # evaluation output directory
test_data_ground = "/projects/shart/digital_pathology/data/PenMarking/testing_data_groundTruth"

img_left_area = (0, 0, 256, 256)
img_right_area = (256, 0, 512, 256)

testing_cases = open(testing_case_txt_50, 'r').readlines()
for tc in testing_cases:
    uuid, x = os.path.splitext(os.path.split(tc)[1])
    print("processing %s" % uuid)
    img_fn_list = os.listdir(os.path.join(patch_root_dir, uuid))
    for img_fn in img_fn_list:
        img = Image.open(os.path.join(patch_root_dir, uuid, img_fn))
        img_left = img.crop(img_left_area)
        img_right = img.crop(img_right_area)

        save_to = os.path.join(test_data, uuid)
        if not os.path.exists(save_to):
            os.makedirs(save_to)
        img_right.save(os.path.join(test_data, uuid, img_fn))  # inputs
        # img.save(os.path.join(save_to, img_fn.replace(".", "_original.")))

        save_to = os.path.join(test_data_ground, uuid)
        if not os.path.exists(save_to):
            os.makedirs(save_to)
        img_left.save(os.path.join(test_data_ground, uuid, img_fn.replace(".", "_targets."))) # targets















