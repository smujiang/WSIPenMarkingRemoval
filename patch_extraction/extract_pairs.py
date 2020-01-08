##################################################
# This code extract image patch pairs from pairwise Whole Slide Images
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

from wsitools.file_management.wsi_case_manager import WSI_CaseManager
from wsitools.file_management.offset_csv_manager import OffsetCSVManager
from wsitools.tissue_detection.tissue_detector import TissueDetector
from wsitools.patch_extraction.pairwise_patch_extractor import PairwiseExtractorParameters, PairwisePatchExtractor
from wsitools.wsi_annotation.region_annotation import AnnotationRegions
import os

wsi_ext = '.tiff'
fixed_wsi_list_txt = "/PenMarking/WSIs/case_list.txt"    # file name list of MARKED WSIs (Dataset for research)
image_pairs_txt = "/PenMarking/WSIs/image_pairs.txt"     # txt file define WSI file name correspondence of image pairs

fixed_wsi_root_dir = "/PenMarking/WSIs/MELF"                      # folder where saves the MARKED WSIs
float_wsi_root_dir = "/PenMarking/WSIs/MELF-Clean"                # folder where saves the CLEAN WSIs

annotation_root_path = "/PenMarking/annotations"                   # folder saves the annotations
class_label_id_csv = "/PenMarking/annotations/class_label_id.csv"  # file in which saves the annotation labels
offset_csv_fn = "/PenMarking/annotations/wsi_pair_offset.csv"      # file in which saves the registration offset

output_dir = "/PenMarking/extracted_patches"

fixed_wsi_list = open(fixed_wsi_list_txt, 'r').readlines()
for fixed_wsi_t in fixed_wsi_list:
    fixed_wsi = os.path.join(fixed_wsi_root_dir, fixed_wsi_t + wsi_ext)
    case_mn = WSI_CaseManager(image_pairs_txt)
    float_wsi = case_mn.get_counterpart_fn(fixed_wsi, float_wsi_root_dir)
    _, fixed_wsi_uuid, _ = case_mn.get_wsi_fn_info(fixed_wsi)
    _, float_wsi_uuid, _ = case_mn.get_wsi_fn_info(float_wsi)

    offset_csv_mn = OffsetCSVManager(offset_csv_fn)
    offset, state_indicator = offset_csv_mn.lookup_table(fixed_wsi_uuid, float_wsi_uuid)
    if state_indicator == 0:
        raise Exception("No corresponding offset can be found in the file")

    xml_fn = os.path.join(annotation_root_path, fixed_wsi_uuid + '.xml')
    tissue_detector = TissueDetector("LAB_Threshold", threshold=80)
    parameters = PairwiseExtractorParameters(output_dir, save_format='.jpg', sample_cnt=-1)
    if os.path.exists(xml_fn):
        annotations = AnnotationRegions(xml_fn, class_label_id_csv)
        patch_extractor = PairwisePatchExtractor(tissue_detector, parameters, annotations=annotations)
        patch_cnt = patch_extractor.extract(fixed_wsi, float_wsi, offset)
    else:
        patch_extractor = PairwisePatchExtractor(tissue_detector, parameters)
        patch_cnt = patch_extractor.extract(fixed_wsi, float_wsi, offset)
    print("%d Patches have been save to %s" % (patch_cnt, output_dir))
