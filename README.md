
#**Traffic Sign Recognition Project Report** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. 

The project code is included in the zip file.

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. 

The code for this step is contained in the code cell #1 and #2 of the IPython notebook.  

I used the collections library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 39209
* The size of test set is 12630
* The shape of a traffic sign image is (32,32)
* The number of unique classes/labels in the data set is 43

I also use the csv library to read in the signnames.csv and store the names in a np.array.

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the code cell #3 of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing how many data sets are included in the training data for each traffic sign.
![Data set numbers for each traffic sign](files/image_01.png)

I also plot a sample of each traffic sign and attach a number on it (representing the frequency in the training set).
![Samples for each traffic sign](files/image_02.png)

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. 

The code for this step is contained in the code cell #4 of the IPython notebook.

I define a function rgb2gray_norm to do the rgb to gray scale change as gray = 0.299*r + 0.587*g + 0.144*b.
And then do a normalization to make the values be between [-0.5,0.5]. The main reason for the normalization is to reduce the numerical error during the optimization process. In addition, keeping the inputs in normalized shapes help the optimizer to find the minimum cost faster.

####2. Describe how, and identify where in your code, you set up training, validation and testing data.

The code for splitting the data into training and validation sets is contained in the fifth code cell of the IPython notebook.  

To cross validate my model, I randomly split the training data into a training set and validation set. I did this by shuffle [x_train,y_train} first and then assign 30% of the training data as the validation data.

My final training set had 27446 number of images. My validation set and test set had 11763 and 12630 number of images.


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) 

The code for my final model is located in the sixth cell of the ipython notebook. 

The architecture used in the final script is a standard LeNet + one drop out layer between the conv layer and the fully connected layer.


####4. Describe how, and identify where in your code, you trained your model. 

The code for training the model is located in the seventh cell of the ipython notebook. 

My model is based on the LeNet architecture and I add in an additional drop out layer. Its structure is shown as follow:


| Layer			        |     Detail	        		    	  | 
|:---------------------:|:---------------------------------------:| 
| Input                 | 32x32x1 (grayscale image input)         | 
| Conv Layer            | 5x5x1 filter, stride 1, output 28x28x6  | 
| Relu		            | 							              |
| Maxpool	      		| 2x2 kernel, 2x2 stride, output 14x14x6  |
| Conv Layer            | 5x5x6 filter, stride 1, output 10x10x16 | 
| Relu		            | 							              |
| Maxpool	      		| 2x2 kernel, 2x2 stride, output 5x5x16	  |
| Flatten    			| output 400 nodes  				      |
| Drop out              | 0.5 keep_prob                           |
| Fully connected layer | 400 nodes in, output 120 nodes          |
| Relu		            | 							              |
| Fully connected layer | 120 nodes in, output 84 nodes           |
| Relu		            | 							              |
| Output layer          | 84 nodes in, output 43 logits           |



During the model training, I use a np.array to record the validation accuracy evolution and plot it after the training is finished. A example is shown as follow.
![Validation accuracy evolution](files/image_03.png)

####5. Describe the approach taken for finding a solution. 

The code for calculating the accuracy of the model is located in the cell #8 of the Ipython notebook.

My final model results were:
* validation set accuracy of 98.9% 
* test set accuracy of 94.7%

I tried the following architectures:

01: Standard LeNet
02: LeNet + drop out
03: LeNet + combine the conv layer output and the first fully connected layer output as the input for the final logistic layer
04: 03 + drop out

The peformance for these four architectures are similar, they both achieve around 99% accuracy on the validation data after around 30 epochs. The 04 method do get to the final accuracy a little bit faster.

The reason I chose LeNet as the base structure is because that LeNet has been proven to do a great job in recognizing english letters, which has similar complexity and similar required resolution as compared to the traffic signs.

The hyperparameters I tried are:


| Hyperparameter	    |     Value     	        		      | 
|:---------------------:|:---------------------------------------:| 
| Batch_size            | 128                                     | 
| Epochs                | 30, 50, 100                             | 
| Learning rate         | 0.001, 0.0002				              |
| Keep_prob	      		| 0.5                                     |

In terms of the number of epochs, thre is no significant difference between 30, 50 and 100. The model achieve around 99% validation accuracy after around 30 epochs and doesn't increase much until epoch 100. So I just use 30 epochs in the final run.

In terms of the learning rate, there is also no significant difference. The model did train slower with the 0.0002 learning rate, i.e., get to 99% accuracy slower. However, the final accuracy after 100 epochs doesn't improve as compared to the 0.001 learning rate results. In the final run, I used 0.001 learning rate.

What confused me is that the test set accuracy is significantly lower than the validation accuracy. Which I assume is due to the difference between the train data set and the test data set.


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are six German traffic signs that I found on the web. In cell #9 and #10, I load the images and perform rgb2gray and normalization on these images and also transform them to 32X32.

The first five images contain single sign while the sixth image contains two signs in one picture. So the final image will be difficult to classify because it has two signs, I am interested to see that what probabilities the classifier will assign to this image.

![1](files/image_04.png)



####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. 

The code for making predictions on my final model is located in the cell #11 of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit 50km/h    | Speed limit 50km/h   						    | 
| Speed limit 30km/h    | Speed limit 30km/h						    |
| Children Crossing		| Children Crossing								|
| No entry	      		| No entry  					 				|
| Bumpy Road			| Bumpy Road      						    	|
| 30km/h limit+children | Priority Road                                 |


The model was able to correctly guess 5 of the first 5 traffic signs, which gives an accuracy of 100%. Consider that the test accuracy of the model is 94.7%, I believe this 100% accuracy on the acquired images is appropriate.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. 

The code for making predictions on my final model is located in the 13th cell of the Ipython notebook.

I plot bar charts as visualizations of the predication certainties. 

![05](files/image_05.png)

For the first image (50 km/h speed limit), the model is very sure that this is a 50km/h speed limit (probability of 98.1%). And the second guess is a 30 km/h speed limit (only 1.9%), I think this is due to the similarity between speed limit signs. The top five soft max probabilities were

               possibility of Speed limit (50km/h) = 0.981
               possibility of Speed limit (30km/h) = 0.019
               possibility of Speed limit (80km/h) = 0.000
               possibility of Speed limit (20km/h) = 0.000
               possibility of Speed limit (70km/h) = 0.000


For the second image (30 km/h speed limit), the model thinks that this is a 30 km/h speed limit (probability of 64.2%). The ratio of the 30 km/h speed limit is changed from 1:1 when I reshape the image to 32X32, and this may be a reason that the probability is only 64.2%. The second guess is a 50 km/h speed limit (only 31%). Again I believe this is due to the similarity between speed limit signs. The top five soft max probabilities were

               possibility of Speed limit (30km/h) = 0.642
               possibility of Speed limit (50km/h) = 0.310
               possibility of Priority road = 0.042
               possibility of Speed limit (100km/h) = 0.004
               possibility of Speed limit (80km/h) = 0.002

For the third image (Children crossing), the model is 100% sure it contains children crossing sign (probability = 100%). The other probabilities are all 0. The top five soft max probabilities were
   
               possibility of Children crossing = 1.000
               possibility of Bicycles crossing = 0.000
               possibility of Right-of-way at the next intersection = 0.000
               possibility of Road narrows on the right = 0.000
               possibility of Dangerous curve to the left = 0.000

For the fourth image (No-entry), the model is 100% sure it contains No-entry sign (probability = 100%). The other probabilities are all 0. he top five soft max probabilities were

               possibility of No entry = 1.000
               possibility of Stop = 0.000
               possibility of No passing = 0.000
               possibility of Roundabout mandatory = 0.000
               possibility of End of all speed and passing limits = 0.000

For the fifth image (bumpy road), the model is 100% sure it contains bumpy road sign (probability = 100%). The other probabilities are all 0. The top five soft max probabilities were

               possibility of Bumpy road = 1.000
               possibility of Turn left ahead = 0.000
               possibility of Slippery road = 0.000
               possibility of Bicycles crossing = 0.000
               possibility of Wild animals crossing = 0.000

For the sixth image that contains both 30 km/h speed limit sign and the children crossing sign, the predication is completely wrong. I think maybe this is due to the smaller size of these two signs on a signle image. By observing the 32X32 version of the sixth image, we can see that the features of these two traffic signs are almost completely lost and even a human can't tell exactly what sign it is. The top five soft max probabilities were

               possibility of Priority road = 0.872
               possibility of Traffic signals = 0.122
               possibility of Right-of-way at the next intersection = 0.004
               possibility of Road work = 0.001
               possibility of Roundabout mandatory = 0.001
