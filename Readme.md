Deep learning models are showing promise in digital pathology to aid diagnoses. 
Training complex models require a significant amount and diversity of well-annotated data, typically housed in institutional archives. 
These slides often contain clinically meaningful markings to indicate regions of interest. 
If slides are scanned with the ink present, then the downstream model may end up looking for regions with ink before making a classification. 
If scanned without the markings, the information is lost about where the relevant regions are located. 

In this repo, we proposed a straightforward framework to digitally remove ink markings from whole slide images using a conditional generative adversarial network, opening the possibility of using archived clinical samples as resources to fuel the next generation of deep learning models for digital pathology.   
Please read our paper to get more details. If you find this repo helps you, please cite our work:
```
@article{jiang2020,
  title={Image-to-Image Translation for Automatic Ink Removal in Whole Slide Images},
}
```

Here is an example of WSI showing the clean scan, marked slides and image after restoration.  
![Thumbnail level](./doc/imgs/sample.png) 
Here are some high resolution image patches to show performance of image restoration .
![High resolution patches](./doc/imgs/sample_patches.png)


## Installation
* Clone our code 
```
git clone https://github.com/smujiang/WSIPenMarkingRemoval.git
```
* Install the dependencies
```
conda install tensorflow-1.14 numpy-1.15 PIL seaborn pandas
```
> You may also need to install our [wsitools]() to enable our patch extraction sub-module.

## Run our workflow
* Step 1 [optional]. Prepare your own dataset   
    Since the volume of image data for this research is huge, we only provide very few image samples in [this directory](./img_samples). 
    
    You may need to extract patches from WSIs, depend on if you would like to run on your own dataset.  
    If you would like to train your own model, you need to prepare some clean an marked image pairs. Please refer to our [pairwise patch extraction code](./patch_extraction/extract_pairs.py)   
    If you would like to test our pre-trained model, you just need marked WSIs for testing. Please refer to our [marked WSI extraction code](./patch_extraction/extract_marked.py)

* Step 2 [optional]. Train the model   
    Please refer to this [bash script](./pix2pix/training.sh) to train the ink removal model. 

* Step 3. Testing the model  
    Please refer to this [bash script](./pix2pix/eval.sh) to evaluate the model. You can train your own model (from step 2), or our pre-trained model which can be downloaded from [Google Drive](https://drive.google.com/open?id=1iFKdA-I2Ms_CzhYY9naHfYAR2xrz4ria).

* Step 4. Reconstruct the image from restored patches  
    Please refer [our code](./post_proc/patch_blending.py) for patch reconstruction 

* Step 5 [optional]. Quantitative evaluation    
    Evaluation metrics can be calculated with code in ./eval

### References
[1] S. Ali, N. K. Alham, C. Verrill, and J. Rittscher, "Ink removal from histopathology whole slide images by combining classification, detection and image generation models," arXiv preprint arXiv:1905.04385, 2019.




