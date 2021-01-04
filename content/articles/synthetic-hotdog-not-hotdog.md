---
title: (Synthetic) Hotdog or not Hotdog?
description: I decided to one-up Jìan-Yáng with my own version of his hotdog-detecting algorithm. My special ingredient? Synthetic Data.
---

Whenever someone decides to go ahead and flex their deep-learning prowess, or they choose to show others a simple and quirky introduction to creating deep computer vision models, it seems like they borrow religiously from the ingenious app creation of Jìan-Yáng. In the famed episode of *Silicon Valley* S4E4, Jìan-Yáng reveals that the revolutionary food app that he's been working on is in fact extremely underwhelming: it can only differentiate between images of hotdogs and images that aren't hotdogs. Bummer. Fortunately, it provides for a fun idea for a simple deep computer vision project that doesn't involve working with the rather boring hand-written digits of the [MNIST Dataset](http://yann.lecun.com/exdb/mnist/) or the abysmal low-resolution images of [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html).

In the spirit of reinventing the wheel and copying the projects of other developers, I decided to create my own hotdog-detecting algorithm. This time, however, there's a special twist: we'll use synthetic data to aid the production process and generate our own data to increase the accuracy of the algorithm.

## Wait, What's Synthetic Data?

Glad you asked. Synthetic data is any data that is generated algorithmically to simulate real data that is hard and/or expensive to collect and label. Synthetic data exists for a variety of niche topics, whether it is [synthetic medical records](https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/s12874-020-00977-1) that allow researchers to hide patients' sensitive data, to [hyperrealistic indoor images](https://github.com/apple/ml-hypersim) that allow researchers to test and train indoor navigational agents. For those of us who don't think that everything needs to be necessarily practical, we can create synthetic hotdog images to train a binary hotdog classifier!

Funny enough, the hotdog not hotdog detector seems to be the perfect case for using synthetic data. The current [dataset](https://www.kaggle.com/dansbecker/hot-dog-not-hot-dog) only has 250 images of hotdogs and 250 images of non-hotdogs. Compare this to a dataset such as [ImageNet](http://image-net.org) with over fourteen million images, and it becomes clear why synthetic data is such a necessity for niche applications where data is difficult to collect.

## How Can I Create Synthetic Data?

Depending on the scope of your real data, there are a few tools available for generating synthetic data. For working with tabular and time-series data, there exists the [Synthetic Data Vault](https://sdv.dev/). Created by MIT researchers, it learns the distribution of tabular and time-series data to create data that is nearly indistinguishable from the original. This is opposed to random data generators such as [Mockaroo](https://mockaroo.com/) that generate randomized data rather than mocking realistic data. 

When it comes to generating visual data, it's a bit harder and requires more domain expertise. To create truly robust synthetic data, you need a 3D model or a 3D scene. Next, you have to incorporate your 3D model/scene into a rendering pipeline to actually generate synthetic images. However, this alone is not enough to generate robust synthetic data, and will ultimately lead to overfitting on a select set of 3D models/scenes. You have to also introduce domain randomization into your scene in order to train a model that doesn't overfit your data. To achieve this, techniques such as random camera placement, Gaussian Blurring, Gaussian Noise, random backgrounds, and random colorization can be applied. The list of domain randomization techniques goes on and on, though the techniques previously mentioned are often some of the most effective for creating a dataset that can train a strong computer vision model. We'll use a subset of these techniques when we go ahead and train our own model.

## Hotdog vs No Hotdog

Here's the fun part where we go ahead and build our synthetic dataset, as well as our computer vision model. We'll use Pytorch to take care of the deep-learning related work for our pipeline, and we'll use Blender to take care of the rendering aspect of our pipeline. More specifically, I'll be using a simple script that I created for generating synthetic data in Blender that you can find [here](). This tool can import Blender objects (.dae files), apply a random uniform rotation to the object, apply a random background, Gaussian Noise, and Gaussian Blur, and then render the final image. You can find the 3D .dae files [here](), and the background images that I used [here](). 

Next, we'll create a simple binary classifier in Pytorch to train on our synthetic data. In my case, I created the binary classifier in Google Colab because my GTX 1650 isn't necessarily purpose-built for training computer vision models, and I'm too broke to rent out a Tesla v100 or P100 Cloud GPU.

### Creating our Data 
First, we have to create our synthetic data. As mentioned above, we'll go ahead and import our 3D objects as well as our background images. I went ahead and downloaded twenty images to use as backgrounds from Flickr: ten images of tables (you gotta eat somewhere) and ten images of ballparks (when else would a normal person eat hotdogs?). Next, I went ahead and found some 3D hotdogs on [Sketchfab](https://sketchfab.com/search?q=tag%3Ahotdog&sort_by=-relevance&type=models). One was low-poly and one was just a pig in a blanket, but I'm too cheap to go ahead and purchase the professionally created 3D hotdog models. Once I had these, I made sure to convert my 3D objects into the .dae format, and I went ahead and put them in the proper directory for the Synthblend script. Finally, I went ahead and rendered 5,000 images of hotdogs because why not.

Please note that all 3D objects and images used in making our synthetic dataset have a creative commons license, though I don't plan on creating a startup with my hotdog detecting algorithm anytime soon (that niche has already been taken care of, unfortunately.) You can find the backgrounds and the objects in their respective folders at [this]() repository.

### Importing our Data
Next, we'll import the necessary Pytorch packages to create our model. We'll need the base ```torch``` and ```torchvision``` libraries. We also need to load in our data from two separate folders: one containing images of hotdogs, and another containing images of food that isn't a hotdog.
```python
import torch
import torch.nn as nn
from torch.utils.data import dataloader

from torchvision import datasets, transforms
```
Next, we'll go ahead and define our data transforms. We won't get fancy with the augmentations, since the domain randomization techniques applied when the synthetic data was created should suffice. We'll simply crop each image to be 128 by 128 pixels, and then normalize each pixel within the range of -1 and 1.
```python
transform = transforms.Compose([
    transforms.Resize(128),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
```
Finally, we'll go ahead and load in our data using the ```torchvision``` ImageFolder class. We'll extend this class such that we can create a training and testing split to our data. We won't use a validation set in our data since we won't be performing any hyperparameter tuning, and we'll arbitrarily split our data as 80% training and 20% testing. We'll also go ahead and download our dataset from [Kaggle](https://www.kaggle.com/dansbecker/hot-dog-not-hot-dog), keeping the directory structure to work with our ImageFolder class

### Creating our Model
For all intents and purposes, we'll use a simple network with residual blocks and average pooling. Could we implement a State-of-the-Art model such as [EfficientNet](https://arxiv.org/abs/1905.11946) with State-of-the-Art activation functions such as the [Funnel Rectified Linear Unit](https://arxiv.org/abs/2007.11824)? Of course, we could! Is that way beyond the scope of an article discussing detecting hotdogs? 100%.
```python

```

### Creating the Training Loop
Now that we have all the building blocks in place, it's time that we bring it all together and actually train our model. We'll go ahead and use Tensorboard so that we don't have to scrutinize our terminal to view the loss every few batches. We'll go ahead and use the ADAM optimizer and train our model for fifty epochs.

