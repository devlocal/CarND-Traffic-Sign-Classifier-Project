# Traffic Sign Recognition Project. Writeup.

## Introduction

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./img/exploratory_vis.png "Visualization"
[image2]: ./img/reduced.png "Reduced number of samples"
[image3]: ./img/sign_samples.png "Traffic sign examples"
[image4]: ./img/experiments1.png "LeNet architecture"
[image5]: ./img/experiments2.png "Different network architectures"
[image6]: ./img/experiments3.png "Different batch size"
[image7]: ./img/experiments4.png "Different learning rate"
[image8]: ./img/experiments5.png "Different data sets"
[image9]: ./img/experiments6.png "Dropout layer"
[image10]: ./img/f1.png "F1 score"
[image11]: ./img/new_signs.png "New signs"
[image12]: ./img/layer1c.png "Visualization layer 1 convolution"
[image13]: ./img/layer1a.png "Visualization layer 1 activation"
[image14]: ./img/layer1p.png "Visualization layer 1 pooling"
[image15]: ./img/layer2c.png "Visualization layer 2 convolution"
[image16]: ./img/layer2a.png "Visualization layer 2 activation"
[image17]: ./img/layer3c.png "Visualization layer 3 convolution"
[image18]: ./img/layer3a.png "Visualization layer 3 activation"
[image19]: ./img/layer3p.png "Visualization layer 3 pooling"
[image20]: ./img/train_samples.png "Training image samples"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---

## Data Set Summary & Exploration

I used the `shape` function of numpy array to get the summary of the traffic signs data set:

* The size of training set is 34799 examples;
* The size of the validation set is 4410 examples;
* The size of test set is 12630 examples;
* The shape of a traffic sign image is 32 x 32 x 3;
* The number of unique classes/labels in the data set is 43.

### Exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is subset of images from a training data set and a bar chart showing how the data is distributed across classes, ordered by number of examples in each class. As can be seen from the chart, the number of examples is different in different classes, with about an order of magnitude difference between the biggest and the smallest class.

![alt text][image20]

![alt text][image1]

## Designing and Testing a Model Architecture

### Data augmentation and pre-processing

I decided to generate additional data samples by:
1. adding random noise to each pixel in all three of RGB channels of an image;
1. multiplying values of RGB channels by the same random factor for each channel, i.e. changing image brightness. I use an average brightness threshold of 150, images that are darker than the threshold are brightened, images that are brighter than the threshold are darkened;
1. enlarging images to the size of 40x40 pixels and then cropping the central 32x32 part, i.e. zooming in by 25%.

I combined original and modified data into the following data sets:

| Data Set         	     | Description                                                    | Size          | 
|:----------------------:|:--------------------------------------------------------------:|:-------------:| 
| `X_train, y_train`     | Original data set                                              | ~35K samples  |
| `X_train_2, y_train_2` | original + images with added noise                             | ~70K samples  |
| `X_train_3, y_train_3` | original + noise + data set with scaled colors                 | ~104K samples |
| `X_train_4, y_train_4` | original + noise + scaled colors + enlarged images             | ~139K samples |
| `X_train_5, y_train_5` | original + noise + enlarged images                             | ~104K samples |
| `X_train_6, y_train_6` | a copy of the previous data set with reduced number of samples | ~68K samples  |

The reason of reducing data set size in `X_train_6, y_train_6` is to reduce the difference in number of samples in each class. The bar chart in the "Exploratory visualization of the dataset" section has a horizontal line at the level of 800 samples. I randomly removed samples from classes where their number is more than 800 to bring the number of samples down to make the distribution more even across classes.

![alt text][image2]

*Note: since I have more data in `X_train_6, y_train_6` (original + noise + zoom in) number of samples in biggest classes is 3 times more than the threshold which is 2400 samples per class.*

Here is an example of an original image and augmented images with noise, colors and an enlarged image.

![alt text][image3]

The only pre-processing step I applied to each of training, validation and test data sets is normalizing data by scaling features `X = (X - 128) / 128`.

I decided not to convert images to grayscale as I believe color is an important feature of a traffic sign. Converting to grayscale could reduce the data set size by a factor of 3 (since only one channel will be used to encode image data as opposed to 3 for RGB images), and could increase training speed. I assume this would decrease accuracy though, although this hypothesis needs to be proved.

## Model architecture.

I designed a set of builder function to be able to experiment with different network architectures defined by metadata rather than by code. Using this toolset, I ran series of experiments.

I tried original LeNet architecture, and several networks with increased number of convolutional layers, fully-connected layers and both convolutional and FC layers.

## Training

I used Adam optimizer to train models. Default value of learning rate was set to 0.001, default batch size — to 128 samples. Default number of epochs was set to 20.

I used validation accuracy as a metric to compare performance of different experiments.

### Experiment series 1 (LeNet architecture)

The first pair of network architectures I experimented with are original LeNet hardcoded in the notebook and an identical architecture defined by `LENET_TOPOLOGY` symbol.

Original LeNet architecture and architecture of `LENET_TOPOLOGY`:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				    |
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16                   |
| Fully connected       | inputs 400, outputs 120                       |
| RELU					|												|
| Fully connected       | outputs 84                                    |
| RELU					|												|
| Fully connected       | outputs 43                                    |

I ran training on both networks with the original data set and got similar results, which is expected. The chart below illustrates validation accuracy, detailed charts are available in the notebook.

![alt text][image4]

Validation accuracy after 20 epochs is **0.914** for LeNet and **0.910** for dynamically built network.

### Experiment series 2 (different network architectures)

Net next series of experiments was to try different network architectures.  I decided to increase the number of convolutional layers for one of the new networks, increase the number of dense layers for another one, and increase the number of both convolutional and dense layers for the third network.

The first network is defined by `NETWORK1` symbol:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 30x30x9 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 15x15x9 				    |
| Convolution 4x4     	| 1x1 stride, valid padding, outputs 12x12x20 	|
| RELU					|												|
| Convolution 4x4     	| 1x1 stride, valid padding, outputs 8x8x40 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 4x4x40                   |
| Fully connected       | inputs 640, outputs 260                       |
| RELU					|												|
| Fully connected       | outputs 130                                   |
| RELU					|												|
| Fully connected       | outputs 43                                    |

The second network is defined by `NETWORK2` symbol: 

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				    |
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16                   |
| Fully connected       | inputs 400, outputs 300                       |
| RELU					|												|
| Fully connected       | outputs 200                                   |
| RELU					|												|
| Fully connected       | outputs 120                                   |
| RELU					|												|
| Fully connected       | outputs 43                                    |

The second network is defined by `NETWORK3` symbol: 

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 30x30x9 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 15x15x9 				    |
| Convolution 4x4     	| 1x1 stride, valid padding, outputs 12x12x20 	|
| RELU					|												|
| Convolution 4x4     	| 1x1 stride, valid padding, outputs 8x8x40 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 4x4x40                   |
| Fully connected       | inputs 640, outputs 400                       |
| RELU					|												|
| Fully connected       | outputs 200                                   |
| RELU					|												|
| Fully connected       | outputs 100                                   |
| RELU					|												|
| Fully connected       | outputs 43                                    |

Experiment results are summarized in the chart below which shows validation accuracy for different network architectures.

Since these networks are larger than the original LeNet network, I decided to run training for 40 epochs.

The network defined by symbol `NETWORK1` (with the same number of fully-connected layers as LeNet, and one additional convolutional layer) produces the best results. It has converged by epoch 30-32 with validation accuracy **0.974**.

Network defined by symbol `NETWORK2` (with the same set of convolutional layers as LeNet, but with an additional fully-connected layer) has also converged by the same number of epochs, however it's final validation accuracy is lower, **0.951**.

Finally, network defined by symbol `NETWORK3` (with an additional convolutional and an additional fully-connected layer) was not able to stabilize, and it's accuracy is somewhere between the first and the second networks (the final value is **0.956**).

I have picked `NETWORK1` for my further experiments.

![alt text][image5]

### Experiment series 3 (different batch size)

Default batch size has been set to 128, I have tried batch sizes 10, 30, 300 and 1000. Results are summarized in the chart below which shows validation accuracy for different batch sizes.

It turns out that the default batch size 128 produces the best validation accuracy.

![alt text][image6]

### Experiment series 4 (different learning rate)

Default learning rate has been set to 0.001. I have tried learning rates of 0.01, 0.003, 0.0003 and 0.0001. The results are summarized in the chart below.

There is no improvement compared to the default learning rate 0.001.

![alt text][image7]

### Experiment series 5 (different data sets)

I have tried to run training using different data sets. It was surprising, but neither of the data sets was able to beat performance of `NETWORK1` with the original data set.

Final values of validation accuracy are:

| Data set         	     | Accuracy	  | 
|:----------------------:|:----------:| 
| `X_train_2, y_train_2` | 0.962      | 
| `X_train_3, y_train_3` | 0.960      | 
| `X_train_4, y_train_4` | 0.963      | 
| `X_train_5, y_train_5` | 0.964      | 
| `X_train_6, y_train_6` | 0.958      | 

![alt text][image8]

#### A follow-up to experiment series 5

As a follow-up to experiment series 5 I ran training of the same network with 60 epochs to see if the network is able to learn better (as it is exposed to more data) with time. Increased number of epochs gave no improvement over already established winner. 

I also tried to train `NETWORK3` on data sets from experiment 5, assuming that a larger network will be able to better learn increased data sets. Larger network was not able to produce an increase of validation accuracy.

Detailed results of these experiments can be seen in the notebook.

#### Experiment series 6 (trying dropout layer)

The best result achieved so far is training `NETWORK1` on the original data set. Validation accuracy is 0.974, training accuracy is 1.0. High training accuracy indicates that the network is probably overfitting.

I decided to try adding a dropout layer to the architecture of `NETWORK1` to prevent overfitting. I've added one dropout layer after the first convolutional layer with *keep probability* 0.5 as `NETWORK4` and *keep probability* 0.75 as `NETWORK5`.

Networks with a dropout layer do not produce better validation set accuracy than `NETWORK1`. Validation accuracy is 0.922 for `NETWORK4`, and 0.920 for `NETWORK5`.

![alt text][image9]

## Conclusion of training experiments

Selected network architecture is `NETWORK1` which has been trained as part of experiment series 2 for 40 epochs on the original data set with default values of hyper-parameters.

My final model results were:
* training set accuracy of 1.0
* validation set accuracy of 0.974
* test set accuracy of 0.959

I have computed precision, recall and F1 score for the test data set. F1 score is shown on the graph below for each class along with class size. As seen from the graph, F1 score is lower for classes with a higher number of examples. Perhaps network performance can be further improved by collecting more training data.

![alt text][image10]

## Testing selected Model on New Images

I have downloaded multiple images of german traffic signs, stored them in the `map_data` folder and annotated each image with coordinates of traffic sign(s) in the image in the `STREET_VIEW_IMAGES` dictionary. I have also manually labeled each image in the field `label` of the same dictionary.

The network is able to correctly classify most of the images. I picked 5 images for further analysis: 4 images that the network is able to classify and 1 image that the network has problems with.

### New image summary

Here are 5 of German traffic signs that I found on the web:

![alt text][image11]

All the images have good image quality and I don't expect difficulties with their classification except probably the second image "No entry" because it was captured from an angle and therefore is distorted: its width is noticeably smaller than its height.

### Model's predictions on the new traffic signs

Here are the results of the prediction:

| Image                                 | Prediction                                    | 
|:-------------------------------------:|:---------------------------------------------:| 
| Traffic signals                       | Traffic signals                               | 
| No entry                              | No entry                                      |
| Speed limit (30km/h)                  | Speed limit (30km/h)                          |
| Right-of-way at the next intersection | Right-of-way at the next intersection         |
| Bicycles crossing                     | Slippery Road                                 |

As seen, the model had no problems classifying the second "No entry" sign, but incorrectly classified "Bicycles crossing" as "Slippery road".

The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. Accuracy over the whole set of 39 new images is 0.974.

The accuracy on the set of new images (0.974) is slightly better than the accuracy on the test set (0.959). I think the reason of that is that new images have a higher image quality than the images on the test set (a subset of test images is rendered in the notebook).

The model is very confident in its predictions for the first 4 images with softmax probability very close to 1.

For the last image "Bicycles crossing", the model incorrectly predicted "Slippery road" with softmax probability of 0.963. The next class is the correct one "Bicycles crossing" with  softmax probability of 0.029.

Probable reason of mistake is that the "Bicycles crossing" class (class id=29) has a relatively low number of training examples.

Detailed report can be seen in the notebook.

## Visualizing the Neural Network

Below is the visualization of the network feature maps for one of the images. Layer 1 of the neural network captures small features of images, such as edges. Layer 2 captures higher-level feature such as shapes. It is hard to interpret visualization of layers 3.

Layer 1 convolution:

![alt text][image12]

Layer 1 activation:

![alt text][image13]

Layer 1 pooling:

![alt text][image14]

Layer 2 convolution:

![alt text][image15]

Layer 2 activation:

![alt text][image16]

Layer 3 convolution:

![alt text][image17]

Layer 3 activation:

![alt text][image18]

Layer 3 pooling:

![alt text][image19]
