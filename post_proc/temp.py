import os
p = '../img_samples'
f_list = os.listdir(p)
for img_fn in f_list:
    new_img_fn = img_fn.replace(", ", "_")
    os.rename(os.path.join(p, img_fn), os.path.join(p, new_img_fn))











