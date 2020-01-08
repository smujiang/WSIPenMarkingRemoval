# from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import openslide
from xlrd import open_workbook
import numpy as np
from PIL import Image
from natsort import natsorted
from sklearn.naive_bayes import GaussianNB
import sys,os
import metrikz
import multiprocessing

def calculate_metrics(Img_a, Img_b):
    PSNR = metrikz.psnr(Img_a, Img_b)
    SSIM = metrikz.ssim(Img_a, Img_b)
    VIF = metrikz.pbvif(Img_a, Img_b)
    # print("A: PSNR: %f, SSIM: %f, VIF: %f" % (PSNR, SSIM, VIF))
    return PSNR, SSIM, VIF
    # PSNR = psnr(Img_a, Img_b)
    # SSIM = ssim(Img_a, Img_b)
    # VIF = vifp(Img_a, Img_b)
    # print("B: PSNR: %f, SSIM: %f, VIF: %f" % (PSNR, SSIM[0], VIF))
    # return PSNR, SSIM[0], VIF


def parse_all_fn(img_file_name):
    ele = img_file_name.split("_", maxsplit=5)
    marked_wsi_uuid, marked_loc_x, marked_loc_y, clean_wsi_uuid, clean_loc_x, others = ele
    clean_loc_y, label_txt, ito = others.split("-")
    fn_type = ito.split(".")[0]
    return marked_wsi_uuid, marked_loc_x, marked_loc_y, clean_wsi_uuid, clean_loc_x, clean_loc_y, fn_type


def get_pairwise_fn(input_path_name):
    path_ele = os.path.split(input_path_name)
    ele = path_ele[1].rsplit("-", maxsplit=1)
    sufix, others = ele
    fn_type = others.split(".")[0]
    if fn_type == "inputs":
        target_fn = os.path.join(path_ele[0], sufix + "-targets." + others.split(".")[1])
        output_fn = os.path.join(path_ele[0], sufix + "-outputs." + others.split(".")[1])
        if os.path.exists(target_fn) and os.path.exists(output_fn):
            return target_fn, output_fn
        else:
            print(target_fn)
            print(output_fn)
            raise Exception("Target or output file missing")
    else:
        raise Exception("Input parameter should be inputs")


def get_all_val_fns(data_dir):
    val_file_names = []
    img_fns = os.listdir(data_dir)
    print("There are %d samples to be calculated" % (len(img_fns) / 3))
    for fn in img_fns:
        marked_wsi_uuid, marked_loc_x, marked_loc_y, clean_wsi_uuid, clean_loc_x, clean_loc_y, fn_type = parse_all_fn(fn)
        if "inputs" == fn_type:
            input_fn = os.path.join(data_dir, fn)
            target_fn, output_fn = get_pairwise_fn(input_fn)
            val_file_names.append([input_fn, output_fn, target_fn])
    return val_file_names


def init_batch_eval():
    global cnt
    cnt = 0


def cal_eval(eval_files):
    input_fn, output_fn, target_fn = eval_files
    input_patch = Image.open(input_fn)
    restored_patch = Image.open(output_fn)
    clean_patch = Image.open(target_fn)
    PSNR, SSIM, VIF = calculate_metrics(np.array(clean_patch), np.array(input_patch))
    PSNR_r, SSIM_r, VIF_r = calculate_metrics(np.array(clean_patch), np.array(restored_patch))
    marked_wsi_uuid, marked_loc_x, marked_loc_y, clean_wsi_uuid, clean_loc_x, clean_loc_y, _ = parse_all_fn(input_fn)
    # wrt_str = marked_wsi_uuid + "," + str(marked_loc_x) + "," + str(marked_loc_y) + "," + clean_wsi_uuid + "," + str(clean_loc_x) + "," + str(clean_loc_y)
    # wrt_str += "," + str(PSNR) + "," + str(PSNR_r) + "," + str(SSIM) + "," + str(SSIM_r) + "," + str(VIF) + "," + str(VIF_r) + "\n"
    wrt_str = input_fn + "," + str(PSNR) + "," + str(PSNR_r) + "," + str(SSIM) + "," + str(SSIM_r) + "," + str(VIF) + "," + str(VIF_r) + "\n"
    metric_arr = [PSNR, PSNR_r, SSIM, SSIM_r, VIF, VIF_r]
    print(wrt_str)
    print(metric_arr)
    return wrt_str, metric_arr


if __name__ == "__main__":
    data_dir = "/data/PenMarking/eval/pixel2pixel_256_50cases/images"
    metrics_csv = "/data/PenMarking/eval/pixel2pixel_256/metrics/psnr_ssim_vif_50.csv"
    bufsize = 1  # 0 means unbuffered, 1 means line buffered,
    metrics_csv_fp = open(metrics_csv, 'a', buffering=bufsize)
    # wrt_str = "marked_wsi_uuid,marked_loc_x,marked_loc_y,clean_wsi_uuid,clean_loc_x,clean_loc_y,PSNR,PSNR_r,SSIM,SSIM_r,VIF,VIF_r\n"
    wrt_str = "marked_image,PSNR,PSNR_r,SSIM,SSIM_r,VIF,VIF_r\n"
    metrics_csv_fp.write(wrt_str)

    val_file_names = get_all_val_fns(data_dir)
    metric_arr_stack = np.empty([1, 6])

    multiprocessing.set_start_method('spawn')
    pool = multiprocessing.Pool(processes=5, initializer=init_batch_eval)
    for wrt_str, metric_arr in pool.imap(cal_eval, val_file_names):
        metrics_csv_fp.write(wrt_str)
        metric_arr_stack = np.vstack([metric_arr_stack, metric_arr])

    metrics_csv_fp.close()
    print("Before/After ink removal, mean_PSNR, mean_SSIM, mean_VIF")
    print(np.mean(metric_arr_stack, axis=0))







