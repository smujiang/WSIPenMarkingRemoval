Deep learning models are showing promise in digital pathology to aid diagnoses. 
Training complex models require a significant amount and diversity of well-annotated data, typically housed in institutional archives. 
These slides often contain clinically meaningful markings to indicate regions of interest. 
If slides are scanned with the ink present, then the downstream model may end up looking for regions with ink before making a classification. 
If scanned without the markings, the information is lost about where the relevant regions are located. 

In this repo, we proposed a straightforward framework to digitally remove ink markings from whole slide images using a conditional generative adversarial network, opening the possibility of using archived clinical samples as resources to fuel the next generation of deep learning models for digital pathology.
![alt text](./doc/imgs/sample.png) 
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








