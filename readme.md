# Introduction
Deep learning in histopathology has developed an interest over the decade due to its improvements in classification and localization tasks. Breast cancer is a prominent cause of death in women. 


Computer-Aided Pathology is essential to analyze microscopic histopathology images for diagnosis with an
increasing number of breast cancer patients. The convolutional neural network, a deep learning algorithm, provides significant results in classification among cancer and non-cancer tissue images but lacks in providing interpretation. Here in this blog, I am writing the streamlined version of the paper **"Breast cancer histopathology image classification and localization using multiple instance learning"** published in **WIECON-2019** in which we have aimed to provide a better interpretation of classification results by providing localization on microscopic histopathology images. We frame the image classification problem as weakly supervised multiple instance learning problems and use attention on instances to localize the tumour and normal regions in an image. **Attention-based multiple instance learning (A-MIL)** is applied on **BreakHis** and **BACH** datasets. The classification and visualization results are compared with other recent techniques. A method used in this paper produces better localization results without compromising classification accuracy.

# About Grad-CAM Image Visualizations

In the era of deep learning, understanding of the model's decision is important and the GradCAM is one of the first and good methods to visualize the outcome. Here is the paper and the following are the results taken from paper directly.

<!--       <center> ![hey](/images/amil/grad_cam.png) </center>
      <center>This is an image</center>
 -->
   	
![Highlighting the part in the input image responsible for classification in that category. Image is taken from the paper directly.](https://dipeshtamboli.github.io/images/amil/grad_cam.png)

Here, in the image, you can see the highlighted portion corresponding to the parts of the image which is responsible for the classification. Like in the first image of Torch, GradCAM is highlighting the portion in the image where the torch is present. Similarly, in the Car Mirror image, it's highlighting the portion where the car mirror is present. Thus this explains the reason behind the decision taken by the model.

![This is the image where the model's output is the cat and GradCAM is highlighting the portion responsible for that decision. Image is taken from the paper directly.](https://dipeshtamboli.github.io/images/amil/dog.png)

This is the image where the model's output is the cat and GradCAM is highlighting the portion responsible for that decision. Image is taken from the paper directly.Now, these are the results where we see that the GradCAM and Guided GradCAM gives us the portion which is important for decision making. But it doesn't work well on the medical images(especially Histopathology images).



![GradCAM and Guided-GradCAM is not highlighting the useful portion of the image.](https://dipeshtamboli.github.io/images/amil/grad_amil.png)

So we have proposed another visualization technique for it. It's an attention-based visualization method where we are doing multiple instance learning.

# Attention-based multiple instance learning (A-MIL)

In this method, we have cropped an image in small square patches and made a bag of it. This bag of images will act like a batch. We are fee
AMIL ArchitectureFirst making a bag of the input image by taking the small patches from it.
Passing it to the feature extractor which is basically a convolutional neural network block.
Then we are passing the Instance level features to the classifier for getting Instance level attention.
Here we are getting the attention weights which we are further using for attention aggregation to get the bag level features.

![The figure shows localization in sample malignant image using Attention - Multiple Instance Learning. A-MIL accurately highlights affected gland and ignores background region](https://dipeshtamboli.github.io/images/amil/amil_arc.png)


Then we are applying Dense layer for the classification of the Benign, Malignant or Invasiveconsidering the

So in the end, we have cropped image patches and their attention weights. We multiplied each attention weight with the corresponding patch and stitch the whole image to get the visualization of the complete input image. With this method, we are neither compromising the accuracy nor made the model complicated. This is just adding transparency to the whole process.

![The figure shows localization in sample malignant image using Attention - Multiple Instance Learning. A-MIL accurately highlights affected gland and ignores background region](https://dipeshtamboli.github.io/images/amil/amil.png)


This is the comparison between the visualization of GradCAM and AMIL method. Here we cropped two portions from the image which is important for the classification and applied GradCAM on it. In another scene, AMIL visualization is there which is properly highlighting the useful portion.
Comparison of the visualization output of GradCAM and A-MILAnother result of AMIL visualization of BACH image.


![Another result from the BACH dataset.](https://dipeshtamboli.github.io/images/amil/result.png)

***********************

[Same article is also available on Medium](https://medium.com/@dipeshtamboli/a-sota-method-for-visualization-of-histopathology-images-1cc6cc3b76f3)   
And also on [my blogs](https://dipeshtamboli.github.io/blog/2019/Visualization-of-Histopathology-images/)

<button style="background-color:azure;color:white;width:200px;
height:40px;">[Arxiv link](https://arxiv.org/abs/2003.00823)</button> | <button style="background-color:azure;color:white;width:240px;
height:40px;">[IEEE Xplore link](https://ieeexplore.ieee.org/abstract/document/9019916)</button> | <button style="background-color:azure;color:white;width:300px;
height:40px;">[Harvard digital library link](https://ui.adsabs.harvard.edu/abs/2020arXiv200300823P/abstract)</button>

# How to run the code

Here, in the folder AMIL_project, we have following folders:

## my_network
	-In this, we have trained a model using this dataset on our own architecture. Accuracies are comparable to the VGG_pretrained and ResNet_pretrained model.
	-In this folder, we have
		-net.py
			-use "python net.py" to run the code(without quotes)
			-this is the code which will train the model, and test it on the validation set
			-this code will save following things in corresponding zoom level folder
				-model(in pickel and pytorch format)
				-terminal logs
				-tensorboard run logs
				-text file summarizing the run
		-run_for_all_zoom.sh
			- use "bash run_for_all_zoom.sh" to run this script(without quotes)
			- this script will run vgg_pre.py for all the zoom level and for all the epochs
			- u can keep the number of epoch = 10	
## ResNet
	-In this, we have used a pre-trained ResNet model and trained last fully connected layer using this dataset.
	-In this folder, we have
		-resnet_pre.py
			-use "python resnet_pre.py" to run the code(without quotes)
			-this is the code which will train the model, and test it on the validation set
			-this code will save following things in corresponding zoom level folder
				-model(in pickel and pytorch format)
				-terminal logs
				-tensorboard run logs
				-text file summarizing the run
		-run_for_all_zoom.sh
			- use "bash run_for_all_zoom.sh" to run this script(without quotes)
			- this script will run vgg_pre.py for all the zoom level and for all the epochs
			- u can keep the number of epoch = 10	
## VGG
	-In this, we have used a pre-trained VGG model and trained last fully connected layer using this dataset.
	-In this folder, we have
		-vgg_pre.py
			-use "python vgg_pre.py" to run the code(without quotes)
			-this is the code which will train the model, and test it on the validation set
			-this code will save following things in corresponding zoom level folder
				-model(in pickel and pytorch format)
				-terminal logs
				-tensorboard run logs
				-text file summarizing the run
		-run_for_all_zoom.sh
			- use "bash run_for_all_zoom.sh" to run this script(without quotes)
			- this script will run vgg_pre.py for all the zoom level and for all the epochs
			- u can keep the number of epoch = 10	

## AMIL_codes
	-In this folder, we have
		-amil_model.py
			-it contains attention model(architecture)
		-patch_data.py
			-data loader (takes images as input and crop it to 28*28 and creates a bag)
		-train_n_test.py
			-use "python train_n_test.py" to run the code(without quotes)
			-code which trains the AMIL and then test it on the validation set and saves visualization in the AMIL_visualization folder
			-this code will save following things in corresponding zoom level folder
				-model(pytorch format)
				-terminal logs
				-tensorboard run logs
				-text file summarizing the run			
				-visualization of test images
		-run_for_all_zoom.sh
			- use "bash run_for_all_zoom.sh" to run this script(without quotes)
			- this script will run vgg_pre.py for all the zoom level and for all the epochs
			- u can keep the number of epoch = 20	

## grad_cam
	-In this folder, we have "inputs" folder where you have to put test image
	-In folder "src", in misc_functions.py, on line number 253, you have to put the name of test image
	-run the code "python guided_gradcam.py" without quotes.
	-this will produce resultant visualization images in "results" folder

## Kaggle_Data:
we are using Breakhis dataset from the Kaggle datasets(link to the dataset)   
-download the dataset from the following link:
[Kaggle Dataset](https://www.kaggle.com/kritika397/breast-cancer-dataset-from-breakhis/downloads/fold1.zip/1)  
-rename it to Kaggle_Data   
-We will use this data for Resnet architecture, vgg architecture and mynet architecture   

## AMIL_Data:
	-Here, for attention based multiple instance learning, we will re-arrange the dataset in the given format(readme_data_format.txt)

###	Here, dataset is in this structure:
		fold1
			-test
				-100X
					-B_100X
						-(images)
					-M_100X
						-(images)
				-200X
					-B_200X
						-(images)
					-M_200X
						-(images)
				-400X
					-B_400X
						-(images)
					-M_400X
						-(images)
				-40X
					-B_40X
						-(images)
					-M_40X
						-(images)
			-train
				-100X
					-B_100X
						-(images)
					-M_100X
						-(images)
				-200X
					-B_200X
						-(images)
					-M_200X
						-(images)
				-400X
					-B_400X
						-(images)
					-M_400X
						-(images)
				-40X
					-B_40X
						-(images)
					-M_40X
						-(images)

###	Now, we have to convert it in the following format:
		data_breakhis
			-100X
				-train
					-0
						-images
					-1
						-images
				-test
					-0
						-images
					-1
						-images				
			-200X
				-train
					-0
						-images
					-1
						-images
				-test
					-0
						-images
					-1
						-images							
			-400X
				-train
					-0
						-images
					-1
						-images
				-test
					-0
						-images
					-1
						-images							
			-40X
				-train
					-0
						-images
					-1
						-images
				-test
					-0
						-images
					-1
						-images							
	-rearrange the folders and rename it to AMIL_Data