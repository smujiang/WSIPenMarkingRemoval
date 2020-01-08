
## Installation



## Run our workflow
* Step 1. Prepare your own dataset.   
    If you would like to train your own model, you need to prepare some clean an marked image pairs. Please refer to our [pairwise patch extraction code](./patch_extraction/extract_pairs.py)

    If you would like to test our pre-trained model, you just need marked WSIs for testing. Please refer to our [marked WSI extraction code](./patch_extraction/extract_marked.py)
* Step 2. Train the model[Optional]   
    Please refer to this [bash script](./pix2pix/training.sh) to train the ink removal model. 
* Step 3. Testing   
    Please refer to this [bash script](./pix2pix/eval.sh) to evaluate the model. You can test your own model (from step 2), or our pre-trained model which can be downloaded from [Google Drive](https://drive.google.com/file/d/1kqmhp1IBpJlrY3KObD8O2FOFE4ya7iaG/view?usp=sharing).
* Step 4. Reconstruct the image from restored patches  
    Please refer [our code](./post_proc/patch_blending.py) for patch reconstruction 








