---
title: '[Note] Một số thuật toán diffusion model for super resolution'
date: 2024-02-15
permalink: /posts/2024/02/15/mot-so-thuat-toan-diffusion-model-for-super-resolution/
tags:
  - research
  - proposal
  - diffusion model
--- 

Một số thuật toán diffusion model for super resolution (LDM, ISSR và ESGAN-SwinIR)
 
List of algorithms for super resolution problem
======

1.	LDM algorithm
-	It is one of the models of diffusion model.
-	Folder code: sources/LDM/taming-transformers/
-	Run the main function in main.py file (read carefully the parameters in README.md file)
-	Libraries that need to be installed should note:
o	pip install omegaconf
o	pip install einops
o	pip install pytorch_lightning==1.6.5
o	pip install ipywidgets omegaconf>=2.0.0 pytorch-lightning>=1.0.8 torch-fidelity einops
-	Code implemented in ipynb (LDM.ipynb) for reference on how to run the superresolution model.
-	Link download: https://drive.google.com/drive/folders/1FaBXwa4ScvRc7czDoYGlplRJthPElU-m?usp=sharing

2.	ISSR algoritm (Image-Super-Resolution-via-Iterative-Refinement algorithm)
-	It is one of the models of diffusion model 
-	Folder code: sources/ISSR/Image-Super-Resolution-via-Iterative-Refinement
-	Additional libraries that need to be installed: 
o	pip install lmdb
o	pip install tensorboardx
-	Pretrained model in use: I830000_E32_opt
-	Run file sr.py to train. Run file infer.py to run the results.
-	Read more carefully in the README.md file in the root directory.
-	Code implemented using ipynb (ISSR.ipynb) to see how to infer data. 
-	Link download: https://drive.google.com/drive/folders/1tO17L5RHo3W15JfsQsNSGZyb3g_wYClX

3.	SwinIR and ESRGAN algorithm
-	It is one of the GAN algorithms 
-	Folder code: sources/SwinIR/Real-ESRGAN (root_path)
-	For SwinIR algorithm: root_path/SwinIR/ (swinir_root_path)
-	For ESRGAN, the root folder is root_path (esrgan_root_path)
-	Need to install some additional libraries:
o	!pip install basicsr
o	!pip install facexlib
o	!pip install gfpgan
o	!pip install -r requirements.txt
o	!pip install timm
-	For SwinIR, run file main_test_swinir.py (in folder swinir_root_path). Read more carefully the parameters at swinir_root_path/README.md
-	For ESRGAN model, run file inference_realesrgan.py of folder esrgan_root_path or file inference_realesrgan_video.py (if you want to test on video). You should read the file esrgan_root_path/README.md to better understand the parameters.
-	Pretrained models are saved in esrgan_root_path/pretrained_models
-	Code implemented with ipynb (esrgan_root_path /SwinIR_and_ESRGAN.ipynb) to see how to infer data. 
-	Link download: https://drive.google.com/drive/folders/1Efv3lbSwbdDQvB_9BPQKJD8cwMfdbYqh?usp=sharing
 
Link original source: 
+ https://github.com/JingyunLiang/SwinIR
+ https://github.com/xinntao/ESRGAN
+ https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement
+ https://github.com/IceClear/LDM-SRtuning/tree/main



Hết.
