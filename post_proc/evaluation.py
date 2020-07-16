import os
from PIL import Image
import numpy as np
import sys
sys.path.append(os.path.abspath('../eval'))
from get_overall_eval_metrics import calculate_metrics

wsi_uuid_list = ["7470963d479b4576bc8768b389b1882e", "4e5a6beed06d4ce48be735e1f3c3abc1",
                 "d83cc7d1c941438e93786fc381ab5bb5", "a0e53609686a4ae9a824d9525641dc56",
                 "c477c949f26a40eca92657b1bcf5dcca"]
org_location_1 = [[39824, 40800], [41216, 37908], [41216, 41908], [40216, 38908], [47071, 37706]]
org_location_2 = [[40872, 27687], [42764, 24895], [44208, 26498], [47548, 11087], [45721, 13120]]
org_location_3 = [[21723, 40008], [26019, 26304], [34843, 28973], [68357, 54662], [76221, 58489]]
org_location_4 = [[29220, 49680], [59996, 37196], [61779, 36414], [40985, 59544], [50038, 29307]]
org_location_5 = [[37328, 41051], [21070, 33279], [31096, 24155], [68732, 41287], [38525, 33302]]
org_location_list = [org_location_1, org_location_2, org_location_3, org_location_4, org_location_5]
img_dir = "/projects/shart/digital_pathology/data/PenMarking/eval/pixel2pixel_256/images_dispatch"
img_dir_out = "/projects/shart/digital_pathology/data/PenMarking/eval/pixel2pixel_256/patch_blendings"

for wsi_uuid in wsi_uuid_list:
    print("processing case: %s" % wsi_uuid)
    case_dir = os.path.join(img_dir, wsi_uuid)
    for case_org_locations in org_location_list:
        for org_location in case_org_locations:
            direct_name = os.path.join(img_dir_out, wsi_uuid, "direct_stitch" + str(org_location) + ".jpg")
            direct_rec_img = Image.open(direct_name)
            blending_name = os.path.join(img_dir_out, wsi_uuid, "blending_stitch" + str(org_location) + ".jpg")
            blended_rec_img = Image.open(blending_name)
            org_name = os.path.join(img_dir_out, wsi_uuid, "original_img" + str(org_location) + ".jpg")
            original_img = Image.open(org_name)
            tar_name = os.path.join(img_dir_out, wsi_uuid, "target_img" + str(org_location) + ".jpg")
            target_img = Image.open(tar_name)

            PSNR, SSIM, VIF = calculate_metrics(np.array(target_img), np.array(original_img))
            PSNR_b, SSIM_b, VIF_b = calculate_metrics(np.array(target_img), np.array(blended_rec_img))
            PSNR_s, SSIM_s, VIF_s = calculate_metrics(np.array(target_img), np.array(direct_rec_img))

            print("%f, %f, %f" %(PSNR, SSIM, VIF))
            print("%f, %f, %f" %(PSNR_b, SSIM_b, VIF_b))
            print("%f, %f, %f" %(PSNR_s, SSIM_s, VIF_s))

