import os
import math
import numpy as np

TRAN_NUM = 10

csv_fn = "/data/PenMarking/eval/pixel2pixel_256/metrics/output_%dcases.txt" % TRAN_NUM
psnr_tissue_only = []
ssim_tissue_only = []
vif_tissue_only = []
psnr_ink_only = []
ssim_ink_only = []
vif_ink_only = []
psnr_inked_tissue = []
ssim_inked_tissue = []
vif_inked_tissue = []

psnr_over_all = []
ssim_over_all = []
vif_over_all = []

testing_case_list = ['024aea97fe4f453abb4abef16be7428a',
                    '06ff1caffb3643839d6ca25a037f5ddc',
                    '09ccd9c864194bb2990088348369f291',
                    '0afe6b9fe2d2468e9c5754024757c92f',
                    '0c24709e31e24efbba27406c1ebdb51d',
                    '14ef43b2f4874e17a94271132f48e3e6',
                    '176c3d0c7c6e4807a67d7fc622f1444a',
                    '26caf4fc53b740a0b1845a0ab8c471dc',
                    '2c8f59ea0f584a9ea01eba9b47846579',
                    '307bf553dc894e46ac7b68ef26d22133',
                    '35b2f58edeac4fdcbad85dc0bd83981a',
                    '37427ea3a5ed472f8921bef194fb0ff5',
                    '490ae2ce39e64a29a1d58aa1f865d25f',
                    '4e5a6beed06d4ce48be735e1f3c3abc1',
                    '5bf6b0cf7e874c45a6def5e5360aae64',
                    '5ef732bc2ba54b45a45fa0cdc930380c']
case_list = set()

cnt = 0
lines = open(csv_fn, 'r').readlines()
for l in lines[1:]:
    if l.strip():
        ele = l.strip().split('\t')
        patch_fn = ele[0]
        case_id = os.path.split(patch_fn)[1].split("_", maxsplit=2)[0]
        if case_id in testing_case_list:
            case_list.add(case_id)
            psnr_before = float(ele[1])
            ssim_before = float(ele[2])
            psnr_after = float(ele[4])
            ssim_after = float(ele[5])
            if ele[3] == 'nan' or ele[6] == 'nan':
                pass
            else:
                vif_before = float(ele[3])
                vif_after = float(ele[6])
                psnr_over_all.append([psnr_before, psnr_after])
                ssim_over_all.append([ssim_before, ssim_after])
                vif_over_all.append([vif_before, vif_after])
                if "inked_tissue" in patch_fn:
                    psnr_inked_tissue.append([psnr_before, psnr_after])
                    ssim_inked_tissue.append([ssim_before, ssim_after])
                    vif_inked_tissue.append([vif_before, vif_after])
                elif "tissue_only" in patch_fn:
                    psnr_tissue_only.append([psnr_before, psnr_after])
                    ssim_tissue_only.append([ssim_before, ssim_after])
                    vif_tissue_only.append([vif_before, vif_after])
                elif "ink_only" in patch_fn:
                    psnr_ink_only.append([psnr_before, psnr_after])
                    ssim_ink_only.append([ssim_before, ssim_after])
                    vif_ink_only.append([vif_before, vif_after])
                cnt += 1
                if cnt % 2000 == 0:
                    print("PSNR: %.4f, %.4f; SSIM: %.4f, %.4f; VIF: %.4f, %.4f " % (psnr_before, psnr_after, ssim_before, ssim_after, vif_before, vif_after))

mean_psnr_inked_tissue = np.mean(np.array(psnr_inked_tissue), axis=0)
std_psnr_inked_tissue = np.std(np.array(psnr_inked_tissue), axis=0)
mean_ssim_inked_tissue = np.mean(np.array(ssim_inked_tissue), axis=0)
std_ssim_inked_tissue = np.std(np.array(ssim_inked_tissue), axis=0)
mean_vif_inked_tissue = np.mean(np.array(vif_inked_tissue), axis=0)
std_vif_inked_tissue = np.std(np.array(vif_inked_tissue), axis=0)

mean_psnr_tissue_only = np.mean(np.array(psnr_tissue_only), axis=0)
std_psnr_tissue_only = np.std(np.array(psnr_tissue_only), axis=0)
mean_ssim_tissue_only = np.mean(np.array(ssim_tissue_only), axis=0)
std_ssim_tissue_only = np.std(np.array(ssim_tissue_only), axis=0)
mean_vif_tissue_only = np.mean(np.array(vif_tissue_only), axis=0)
std_vif_tissue_only = np.std(np.array(vif_tissue_only), axis=0)

mean_psnr_ink_only = np.mean(np.array(psnr_ink_only), axis=0)
std_psnr_ink_only = np.std(np.array(psnr_ink_only), axis=0)
mean_ssim_ink_only = np.mean(np.array(ssim_ink_only), axis=0)
std_ssim_ink_only = np.std(np.array(ssim_ink_only), axis=0)
mean_vif_ink_only = np.mean(np.array(vif_ink_only), axis=0)
std_vif_ink_only = np.mean(np.array(vif_ink_only), axis=0)


mean_psnr_over_all = np.mean(np.array(psnr_over_all), axis=0)
std_psnr_over_all = np.std(np.array(psnr_over_all), axis=0)
mean_ssim_over_all = np.mean(np.array(ssim_over_all), axis=0)
std_ssim_over_all = np.std(np.array(ssim_over_all), axis=0)
mean_vif_over_all = np.mean(np.array(vif_over_all), axis=0)
std_vif_over_all = np.std(np.array(vif_over_all), axis=0)

print("mean_psnr_inked_tissue: %f,%f" % (mean_psnr_inked_tissue[0], mean_psnr_inked_tissue[1]))
print("mean_ssim_inked_tissue: %f,%f" % (mean_ssim_inked_tissue[0], mean_ssim_inked_tissue[1]))
print("mean_vif_inked_tissue: %f,%f" % (mean_vif_inked_tissue[0], mean_vif_inked_tissue[1]))

print("mean_psnr_tissue_only: %f,%f" % (mean_psnr_tissue_only[0], mean_psnr_tissue_only[1]))
print("mean_ssim_tissue_only: %f,%f" % (mean_ssim_tissue_only[0], mean_ssim_tissue_only[1]))
print("mean_vif_tissue_only: %f,%f" % (mean_vif_tissue_only[0], mean_vif_tissue_only[1]))

print("mean_psnr_ink_only: %f,%f" % (mean_psnr_ink_only[0], mean_psnr_ink_only[1]))
print("mean_ssim_ink_only: %f,%f" % (mean_ssim_ink_only[0], mean_ssim_ink_only[1]))
print("mean_vif_ink_only: %f,%f" % (mean_vif_ink_only[0], mean_vif_ink_only[1]))

wrt_str = ",PSNR,,SSIM,,VIF,\n"
wrt_str += ",Before,After,Before,After,Before,After\n"

wrt_str += "inked_tissue,%0.2f±%0.2f,%0.2f±%0.2f,%0.2f±%0.2f,%0.2f±%0.2f,%0.2f±%0.2f,%0.2f±%0.2f\n"\
    % (mean_psnr_inked_tissue[0], std_psnr_inked_tissue[0], mean_psnr_inked_tissue[1], std_psnr_inked_tissue[1],
       mean_ssim_inked_tissue[0], std_ssim_inked_tissue[0], mean_ssim_inked_tissue[1], std_ssim_inked_tissue[1],
       mean_vif_inked_tissue[0], std_vif_inked_tissue[0], mean_vif_inked_tissue[1], std_vif_inked_tissue[1])

wrt_str += "tissue_only,%0.2f±%0.2f,%0.2f±%0.2f,%0.2f±%0.2f,%0.2f±%0.2f,%0.2f±%0.2f,%0.2f±%0.2f\n"\
    % (mean_psnr_tissue_only[0], std_psnr_tissue_only[0], mean_psnr_tissue_only[1], std_psnr_tissue_only[1],
       mean_ssim_tissue_only[0], std_ssim_tissue_only[0], mean_ssim_tissue_only[1], std_ssim_tissue_only[1],
       mean_vif_tissue_only[0], std_vif_tissue_only[0], mean_vif_tissue_only[1], std_vif_tissue_only[1])


wrt_str += "ink_only,%0.2f±%0.2f,%0.2f±%0.2f,%0.2f±%0.2f,%0.2f±%0.2f,%0.2f±%0.2f,%0.2f±%0.2f\n"\
    % (mean_psnr_ink_only[0], std_psnr_ink_only[0], mean_psnr_ink_only[1], std_psnr_ink_only[1],
       mean_ssim_ink_only[0], std_ssim_ink_only[0], mean_ssim_ink_only[1], std_ssim_ink_only[1],
       mean_vif_ink_only[0], std_vif_ink_only[0], mean_vif_ink_only[1], std_vif_ink_only[1])

wrt_str += "over_all,%0.2f±%0.2f,%0.2f±%0.2f,%0.2f±%0.2f,%0.2f±%0.2f,%0.2f±%0.2f,%0.2f±%0.2f\n" \
           % (mean_psnr_over_all[0], std_psnr_over_all[0], mean_psnr_over_all[1], std_psnr_over_all[1],
              mean_ssim_over_all[0], std_ssim_over_all[0], mean_ssim_over_all[1], std_ssim_over_all[1],
              mean_vif_over_all[0], std_vif_over_all[0], mean_vif_over_all[1], std_vif_over_all[1])

sv_csv_fn = "/data/PenMarking/eval/pixel2pixel_256/metrics/content_metrics_%d.csv" % TRAN_NUM
fp = open(sv_csv_fn, 'w')
fp.write(wrt_str)

print(case_list)
