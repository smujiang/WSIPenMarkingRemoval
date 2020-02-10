##################################################
# This code extract image patch from marked Whole Slide Images (for testing only)
# Depends on our wsitools in github: https://github.com/smujiang/WSITools
##################################################
# {License_info}
##################################################
# Author: {Jun Jiang}
# Copyright: Copyright {2019}, {project_name}
# Credits: [{Jun Jiang}]
# License: {MIT}
# Version: {1}.{0}.{0}
# Maintainer: {Jun Jiang}
# Email: {Jiang.Jun@mayo.edu}
# Status: {dev}
##################################################

from wsitools.patch_extraction.patch_extractor import ExtractorParameters, PatchExtractor
from wsitools.tissue_detection.tissue_detector import TissueDetector

wsi_fn = "/data/8a26a55a78b947059da4e8c36709a828.tiff" # WSI file name

tissue_detector = TissueDetector("LAB_Threshold", threshold=80)

# extract patches without annotation, no feature map specified and save patches to '.jpg'
output_dir = "/data/wsi_patches"
parameters = ExtractorParameters(output_dir, save_format='.jpg', sample_cnt=-1)
patch_extractor = PatchExtractor(tissue_detector, parameters, feature_map=None, annotations=None)
patch_num = patch_extractor.extract(wsi_fn)
print("%d Patches have been save to %s" % (patch_num, output_dir))


















