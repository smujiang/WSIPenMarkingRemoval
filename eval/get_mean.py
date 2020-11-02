import pandas as pd
import numpy as np

result_file = "/projects/shart/digital_pathology/data/PenMarking_Results/final_cv1.csv"
lines = open(result_file, 'r').readlines()
cnt_sample = 0
PSNR = []
PSNR_r = []
SSIM = []
SSIM_r = []
VIF = []
VIF_r = []
for l in lines:
    if "SSIM" not in l:
        cnt_sample += 1
        if cnt_sample % 1000 == 0:
            print("Processing %d" % cnt_sample)
        ele = l.split(',')
        PSNR.append(float(ele[1]))
        PSNR_r.append(float(ele[2]))
        SSIM.append(float(ele[3]))
        SSIM_r.append(float(ele[4]))
        VIF.append(float(ele[5]))
        VIF_r.append(float(ele[6]))
        if cnt_sample > 200:
            break

m_PSNR = np.array(PSNR).mean(axis=0)
std_PSNR = np.array(PSNR).std(axis=0)
# df_np = pd.read_csv(result_file).to_numpy()
# df_mean = df_np.mean(axis=0)
# df_std = df_np.std(axis=0)
# print(df_mean)
# print(df_std)
print("done")





