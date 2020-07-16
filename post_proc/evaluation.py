import os
from PIL import Image
import numpy as np
import sys
sys.path.append(os.path.abspath('../eval'))
from get_overall_eval_metrics import calculate_metrics

wsi_uuid_list = ["7470963d479b4576bc8768b389b1882e", "4e5a6beed06d4ce48be735e1f3c3abc1",
                  "024aea97fe4f453abb4abef16be7428a", "a0e53609686a4ae9a824d9525641dc56",
                  "c477c949f26a40eca92657b1bcf5dcca"]
org_location_1 = [[39824, 40800], [40216, 38908], [41200, 41200], [40216, 38908]]
org_location_2 = [[39824, 40800], [40216, 38908], [41200, 41200], [40216, 38908]]
org_location_3 = [[39824, 40800], [40216, 38908], [41200, 41200], [40216, 38908]]
org_location_4 = [[39824, 40800], [40216, 38908], [41200, 41200], [40216, 38908]]
org_location_5 = [[39824, 40800], [40216, 38908], [41200, 41200], [40216, 38908]]
org_location_list = [org_location_1, org_location_2, org_location_3, org_location_4, org_location_5]
img_dir = "/projects/shart/digital_pathology/data/PenMarking/eval/pixel2pixel_256/images_dispatch/7470963d479b4576bc8768b389b1882e/"
img_dir_out = "/projects/shart/digital_pathology/data/PenMarking/eval/pixel2pixel_256/patch_blendings"

for wsi_uuid in wsi_uuid_list:
    print("processing case: %s" % wsi_uuid)
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

