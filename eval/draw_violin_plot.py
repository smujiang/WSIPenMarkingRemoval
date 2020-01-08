import os
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

TRAN_NUM = 50

csv_fn = "/data/PenMarking/eval/pixel2pixel_256/metrics/psnr_ssim_vif_%d.csv" % TRAN_NUM
sv_dir = "/data/PenMarking/eval/pixel2pixel_256/metrics"
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

case_list = set()
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

cnt = 0
lines = open(csv_fn, 'r').readlines()
for l in lines[1:]:
    if l.strip():
        ele = l.strip().split(',')
        patch_fn = ele[0]
        case_id = os.path.split(patch_fn)[1].split("_", maxsplit=2)[0]
        if case_id in testing_case_list:
            case_list.add(case_id)
            psnr_before = float(ele[1])
            psnr_after = float(ele[2])
            ssim_before = float(ele[3])
            ssim_after = float(ele[4])
            if ele[5] == 'nan' or ele[6] == 'nan':
                pass
            else:
                vif_before = float(ele[5])
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

delta_psnr = [np.array(psnr_tissue_only)[:,1]-np.array(psnr_tissue_only)[:,0],
              np.array(psnr_inked_tissue)[:,1]-np.array(psnr_inked_tissue)[:,0],
              np.array(psnr_ink_only)[:,1]-np.array(psnr_ink_only)[:,0]]

delta_ssim = [np.array(ssim_tissue_only)[:,1]-np.array(ssim_tissue_only)[:,0],
              np.array(ssim_inked_tissue)[:,1]-np.array(ssim_inked_tissue)[:,0],
              np.array(ssim_ink_only)[:,1]-np.array(ssim_ink_only)[:,0]]

delta_vif = [np.array(vif_tissue_only)[:,1]-np.array(vif_tissue_only)[:,0],
              np.array(vif_inked_tissue)[:,1]-np.array(vif_inked_tissue)[:,0],
              np.array(vif_ink_only)[:,1]-np.array(vif_ink_only)[:,0]]

#########################################################################################
#########################################################################################
psnr_content = []
psnr_diff = []
psnr_values = []
for n in range(len(psnr_tissue_only)):
    psnr_values.append(psnr_tissue_only[n][0])
    psnr_content.append("tissue_only")
    psnr_diff.append("inked")
    psnr_values.append(psnr_tissue_only[n][1])
    psnr_content.append("tissue_only")
    psnr_diff.append("restored")
for n in range(len(psnr_inked_tissue)):
    psnr_values.append(psnr_inked_tissue[n][0])
    psnr_content.append("inked_tissue")
    psnr_diff.append("inked")
    psnr_values.append(psnr_inked_tissue[n][1])
    psnr_content.append("inked_tissue")
    psnr_diff.append("restored")
for n in range(len(psnr_ink_only)):
    psnr_values.append(psnr_ink_only[n][0])
    psnr_content.append("ink_only")
    psnr_diff.append("inked")
    psnr_values.append(psnr_ink_only[n][1])
    psnr_content.append("ink_only")
    psnr_diff.append("restored")

ssim_content = []
ssim_diff = []
ssim_values = []
for n in range(len(ssim_tissue_only)):
    ssim_values.append(ssim_tissue_only[n][0])
    ssim_content.append("tissue_only")
    ssim_diff.append("inked")
    ssim_values.append(ssim_tissue_only[n][1])
    ssim_content.append("tissue_only")
    ssim_diff.append("restored")
for n in range(len(ssim_inked_tissue)):
    ssim_values.append(ssim_inked_tissue[n][0])
    ssim_content.append("inked_tissue")
    ssim_diff.append("inked")
    ssim_values.append(ssim_inked_tissue[n][1])
    ssim_content.append("inked_tissue")
    ssim_diff.append("restored")
for n in range(len(ssim_ink_only)):
    ssim_values.append(ssim_ink_only[n][0])
    ssim_content.append("ink_only")
    ssim_diff.append("inked")
    ssim_values.append(ssim_ink_only[n][1])
    ssim_content.append("ink_only")
    ssim_diff.append("restored")

vif_content = []
vif_diff = []
vif_values = []
for n in range(len(vif_tissue_only)):
    vif_values.append(vif_tissue_only[n][0])
    vif_content.append("tissue_only")
    vif_diff.append("inked")
    vif_values.append(vif_tissue_only[n][1])
    vif_content.append("tissue_only")
    vif_diff.append("restored")
for n in range(len(vif_inked_tissue)):
    vif_values.append(vif_inked_tissue[n][0])
    vif_content.append("inked_tissue")
    vif_diff.append("inked")
    vif_values.append(vif_inked_tissue[n][1])
    vif_content.append("inked_tissue")
    vif_diff.append("restored")
for n in range(len(vif_ink_only)):
    vif_values.append(vif_ink_only[n][0])
    vif_content.append("ink_only")
    vif_diff.append("inked")
    vif_values.append(vif_ink_only[n][1])
    vif_content.append("ink_only")
    vif_diff.append("restored")

######################################
psnr_vio = pd.DataFrame({"metrics": psnr_values,
                         "content": psnr_content,
                         "diff": psnr_diff})
fig = plt.figure(0, figsize=(5, 5))
ax = sns.violinplot(x='content', y='metrics', hue='diff', data=psnr_vio, split=True)
ax.set_title("PSNR")
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles[0:], labels=labels[0:], loc='upper center')  # hide legend title
ax.xaxis.label.set_visible(False)  # hide x label
ax.yaxis.label.set_visible(False)  # hide y label
# plt.show()
# plt.grid(True)
sv_nm = os.path.join(sv_dir, "vio_plot_psnr.png")
fig.savefig(sv_nm, bbox_inches='tight')
plt.close(fig)
######################################
ssim_vio = pd.DataFrame({"metrics": ssim_values,
                         "content": ssim_content,
                         "diff": ssim_diff})
fig = plt.figure(0, figsize=(5, 5))
ax = sns.violinplot(x='content', y='metrics', hue='diff', data=ssim_vio, split=True)
ax.set_title("SSIM")
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles[0:], labels=labels[0:], loc=(0.17, 0.1))  # hide legend title
ax.xaxis.label.set_visible(False)  # hide x label
ax.yaxis.label.set_visible(False)  # hide y label
# plt.grid(True)
sv_nm = os.path.join(sv_dir, "vio_plot_ssim.png")
fig.savefig(sv_nm, bbox_inches='tight')
plt.close(fig)

################################
vif_vio = pd.DataFrame({"metrics": vif_values,
                         "content": vif_content,
                         "diff": vif_diff})
fig = plt.figure(0, figsize=(5, 5))
ax = sns.violinplot(x='content', y='metrics', hue='diff', data=vif_vio, split=True)
ax.set_title("VIF")
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles[0:], labels=labels[0:], loc=(0.2, 0.75))  # hide legend title
ax.xaxis.label.set_visible(False)  # hide x label
ax.yaxis.label.set_visible(False)  # hide y label
# plt.grid(True)
sv_nm = os.path.join(sv_dir, "vio_plot_vif.png")
fig.savefig(sv_nm, bbox_inches='tight')
plt.close(fig)


###################################################################################
colors = ['pink', 'lightblue', 'lightgreen']
fig = plt.figure(0, figsize=(5, 5))
ax = plt.subplot(111)
plt.title("δ-PSNR")
bp = plt.boxplot(delta_psnr, notch=True, patch_artist=True, showfliers=False)
for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
ind = np.arange(1, 5)
plt.xticks(ind, ('tissue_only', 'inked_tissue', 'ink_only'))
plt.grid(True)
sv_nm = os.path.join(sv_dir, "box_plot_delta_psnr.png")
fig.savefig(sv_nm, bbox_inches='tight')
plt.close(fig)
###################################################################################
fig = plt.figure(1, figsize=(5, 5))
ax = plt.subplot(111)
plt.title("δ-SSIM")
bp = plt.boxplot(delta_ssim, notch=True, patch_artist=True, showfliers=False)
for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
ind = np.arange(1, 5)
plt.xticks(ind, ('tissue_only', 'inked_tissue', 'ink_only'))
plt.grid(True)
sv_nm = os.path.join(sv_dir, "box_plot_delta_ssim.png")
fig.savefig(sv_nm, bbox_inches='tight')
plt.close(fig)
###################################################################################
fig = plt.figure(2, figsize=(5, 5))
ax = plt.subplot(111)
plt.title("δ-VIF")
bp = plt.boxplot(delta_vif, notch=True, patch_artist=True, showfliers=False)
for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
ind = np.arange(1, 5)
plt.xticks(ind, ('tissue_only', 'inked_tissue', 'ink_only'))
plt.grid(True)
sv_nm = os.path.join(sv_dir, "box_plot_delta_vif.png")
fig.savefig(sv_nm, bbox_inches='tight')
plt.close(fig)
###################################################################################
###################################################################################
fig = plt.figure(3, figsize=(5, 5))
barWidth = 0.20
bars1 = [mean_psnr_tissue_only[0], mean_psnr_inked_tissue[0], mean_psnr_ink_only[0]]
errs1 = [std_psnr_tissue_only[0], std_psnr_inked_tissue[0], std_psnr_ink_only[0]]
bars2 = [mean_psnr_tissue_only[1], mean_psnr_inked_tissue[1], mean_psnr_ink_only[1]]
errs2 = [std_psnr_tissue_only[1], std_psnr_inked_tissue[1], std_psnr_ink_only[1]]
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
plt.bar(r1, bars1, yerr=errs1, color='blue', width=barWidth, edgecolor='white', label='before', ecolor='black', capsize=5)
plt.bar(r2, bars2, yerr=errs2, color='red', width=barWidth, edgecolor='white', label='after', ecolor='black', capsize=5)
plt.xlabel('PSNR', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(bars1))], ['tissue_only', 'inked_tissue', 'ink_only'])
plt.grid(True)
plt.legend()
sv_nm = os.path.join(sv_dir, "bar_plot_psnr.png")
fig.savefig(sv_nm, bbox_inches='tight')
plt.close(fig)
###################################################################################
fig = plt.figure(4, figsize=(5, 5))
barWidth = 0.20
bars1 = [mean_ssim_tissue_only[0], mean_ssim_inked_tissue[0], mean_ssim_ink_only[0]]
errs1 = [std_ssim_tissue_only[0], std_ssim_inked_tissue[0], std_ssim_ink_only[0]]
bars2 = [mean_ssim_tissue_only[1], mean_ssim_inked_tissue[1], mean_ssim_ink_only[1]]
errs2 = [std_ssim_tissue_only[1], std_ssim_inked_tissue[1], std_ssim_ink_only[1]]
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
plt.bar(r1, bars1, yerr=errs1, color='blue', width=barWidth, edgecolor='white', label='before', ecolor='black', capsize=5)
plt.bar(r2, bars2, yerr=errs2, color='red', width=barWidth, edgecolor='white', label='after', ecolor='black', capsize=5)
plt.xlabel('SSIM', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(bars1))], ['tissue_only', 'inked_tissue', 'ink_only'])
plt.grid(True)
plt.legend()
sv_nm = os.path.join(sv_dir, "bar_plot_ssim.png")
fig.savefig(sv_nm, bbox_inches='tight')
plt.close(fig)
###################################################################################
fig = plt.figure(5, figsize=(5, 5))
barWidth = 0.20
bars1 = [mean_vif_tissue_only[0], mean_vif_inked_tissue[0], mean_vif_ink_only[0]]
errs1 = [std_vif_tissue_only[0], std_vif_inked_tissue[0], std_vif_ink_only[0]]
bars2 = [mean_vif_tissue_only[1], mean_vif_inked_tissue[1], mean_vif_ink_only[1]]
errs2 = [std_vif_tissue_only[1], std_vif_inked_tissue[1], std_vif_ink_only[1]]
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
plt.bar(r1, bars1, yerr=errs1, color='blue', width=barWidth, edgecolor='white', label='before', ecolor='black', capsize=5)
plt.bar(r2, bars2, yerr=errs2, color='red', width=barWidth, edgecolor='white', label='after', ecolor='black', capsize=5)
plt.xlabel('VIF', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(bars1))], ['tissue_only', 'inked_tissue', 'ink_only'])
plt.grid(True)
plt.legend()
sv_nm = os.path.join(sv_dir, "bar_plot_vif.png")
fig.savefig(sv_nm, bbox_inches='tight')
plt.close(fig)

###################################################################################
###################################################################################

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

###################################################################################
ticks = ['tissue_only', 'inked_tissue', 'ink_only']
data_a = [np.array(psnr_tissue_only)[:,0], np.array(psnr_inked_tissue)[:,0], np.array(psnr_ink_only)[:,0]]
data_b = [np.array(psnr_tissue_only)[:,1], np.array(psnr_inked_tissue)[:,1], np.array(psnr_ink_only)[:,1]]
fig = plt.figure(1, figsize=(5, 5))

bpl = plt.boxplot(data_a, positions=np.array(range(len(data_a)))*2.0-0.4, sym='', widths=0.6)
bpr = plt.boxplot(data_b, positions=np.array(range(len(data_b)))*2.0+0.4, sym='', widths=0.6)
set_box_color(bpl, '#D7191C') # colors are from http://colorbrewer2.org/
set_box_color(bpr, '#2C7BB6')

# draw temporary red and blue lines and use them to create a legend
plt.plot([], c='#D7191C', label='before')
plt.plot([], c='#2C7BB6', label='after')
plt.legend()
plt.xticks(range(0, len(ticks) * 2, 2), ticks)
plt.title("PSNR")
plt.grid(True)
plt.tight_layout()
sv_nm = os.path.join(sv_dir, "grouped_box_plot_psnr.png")
fig.savefig(sv_nm, bbox_inches='tight')
plt.close(fig)
###################################################################################
data_a = [np.array(ssim_tissue_only)[:,0], np.array(ssim_inked_tissue)[:,0], np.array(ssim_ink_only)[:,0]]
data_b = [np.array(ssim_tissue_only)[:,1], np.array(ssim_inked_tissue)[:,1], np.array(ssim_ink_only)[:,1]]
fig = plt.figure(1, figsize=(5, 5))

bpl = plt.boxplot(data_a, positions=np.array(range(len(data_a)))*2.0-0.4, sym='', widths=0.6)
bpr = plt.boxplot(data_b, positions=np.array(range(len(data_b)))*2.0+0.4, sym='', widths=0.6)
set_box_color(bpl, '#D7191C') # colors are from http://colorbrewer2.org/
set_box_color(bpr, '#2C7BB6')

# draw temporary red and blue lines and use them to create a legend
plt.plot([], c='#D7191C', label='before')
plt.plot([], c='#2C7BB6', label='after')
plt.legend()
plt.xticks(range(0, len(ticks) * 2, 2), ticks)
plt.title("SSIM")
plt.grid(True)
plt.tight_layout()
sv_nm = os.path.join(sv_dir, "grouped_box_plot_ssim.png")
fig.savefig(sv_nm, bbox_inches='tight')
plt.close(fig)

###################################################################################
data_a = [np.array(vif_tissue_only)[:,0], np.array(vif_inked_tissue)[:,0], np.array(vif_ink_only)[:,0]]
data_b = [np.array(vif_tissue_only)[:,1], np.array(vif_inked_tissue)[:,1], np.array(vif_ink_only)[:,1]]
fig = plt.figure(1, figsize=(5, 5))

bpl = plt.boxplot(data_a, positions=np.array(range(len(data_a)))*2.0-0.4, sym='', widths=0.6)
bpr = plt.boxplot(data_b, positions=np.array(range(len(data_b)))*2.0+0.4, sym='', widths=0.6)
set_box_color(bpl, '#D7191C')  # colors are from http://colorbrewer2.org/
set_box_color(bpr, '#2C7BB6')

# draw temporary red and blue lines and use them to create a legend
plt.plot([], c='#D7191C', label='before')
plt.plot([], c='#2C7BB6', label='after')
plt.legend()
plt.xticks(range(0, len(ticks) * 2, 2), ticks)
plt.title("VIF")
plt.grid(True)
plt.tight_layout()
sv_nm = os.path.join(sv_dir, "grouped_box_plot_vif.png")
fig.savefig(sv_nm, bbox_inches='tight')
plt.close(fig)

