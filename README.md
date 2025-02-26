# **Baybayin OCR**

My final project for my Machine Learning course. Scans Filipino words written in [Baybayin](https://en.wikipedia.org/wiki/Baybayin) script and predicts its Latin transliteration.

## Dataset Development
The dataset was developed using 120 images from various sources. These include printed text, fonts, and handwritten scripts. 
It is then labeled using Label Studio and was structured according to [Ultralytics YOLO dataset](https://docs.ultralytics.com/datasets/detect/) format.

## Model Development
A character recognition model for Baybayin scripts was trained using the dataset. 105 images (87.5%) were used for training and the remaining 15 images (12.5%) for its validation.

Other methods were used to help predict the Filipino text at word-level. 
See the training results and the full paper here:   
[Tagalog Word Prediction of Baybayin Scripts with Optical Character Recognition using YOLO11](https://drive.google.com/file/d/1z3gomVBH_nBzAmOSufL_x4MN2uQHsGmm/view?usp=sharing).
