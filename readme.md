Here, in the folder AMIL_project, we have following folders:
## Kaggle_Data:
we are using Breakhis dataset from the Kaggle datasets(link to the dataset)   
-download the dataset from the following link:   
[Kaggle Dataset](https://www.kaggle.com/kritika397/breast-cancer-dataset-from-breakhis/downloads/fold1.zip/1)  
-rename it to Kaggle_Data   
-We will use this data for Resnet architecture, vgg architecture and mynet architecture   

## AMIL_Data:
	-Here, for attention based multiple instance learning, we will re-arrange the dataset in the given format(readme_data_format.txt)
	-rearrange the folders and rename it to AMIL_Data

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
