---
title: (Synthetic) Hotdog or not Hotdog?
description: I decided to one-up Jìan-Yáng with my own version of his hotdog-detecting algorithm. My special ingredient? Synthetic Data.
---

When a developer decides to go and flex their deep-learning prowess, it seems like they borrow religiously from the ingenious app creation of Jìan-Yáng. In the famed episode of *Silicon Valley* S4E4, Jìan-Yáng reveals that the revolutionary food app that he's been tirelessly working on is in fact extremely underwhelming: it can only differentiate between images of hotdogs and images that aren't hotdogs. Bummer. Fortunately, the concept of *SeeFood* provides for an excellent idea for a simple deep computer vision project that doesn't involve working with the rather banal hand-written digits of the [MNIST Dataset](http://yann.lecun.com/exdb/mnist/) or the abysmal low-resolution images of [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html).

In the spirit of reinventing the wheel and copying personal projects of other developers, I decided to create my own hotdog-detecting binary classification algorithm. This time, however, there's a special twist: we'll use synthetic data to aid the production process and generate our own data to increase the accuracy of the algorithm.

![Hotdog](/article1/hotdog.jpg "The best hotdog to ever exist")
One of the most famous hotdogs | Wired.com

## Wait, What's Synthetic Data?

Glad you asked. Synthetic data is any data that is generated algorithmically to simulate real data that is hard and/or expensive to collect and label. Synthetic data exists for a variety of niche topics, whether it is [synthetic medical records](https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/s12874-020-00977-1) that allow researchers to mask patients' sensitive data, to [hyperrealistic indoor images](https://github.com/apple/ml-hypersim) that allow researchers to test and train indoor navigational agents virtually. For those of us who favor more impractical use cases of deep learning, we can create synthetic hotdog images to train a binary hotdog classifier!

Funny enough, the hotdog, not hotdog detector is an ideal case for using synthetic data. The most widely-accepted hotdog, not hotdog [dataset](https://www.kaggle.com/dansbecker/hot-dog-not-hot-dog) only has 499 images of hotdogs and 499 images of non-hotdogs. Compare this to a dataset such as [ImageNet](http://image-net.org) with over fourteen million images divided between over 21,000 separate classes, and it becomes clear why synthetic data is such a necessity for niche applications where data is difficult to collect.

## How Can I Create Synthetic Data?

Depending on the domain of your real data, there are a few tools available for generating synthetic data. For working with tabular and time-series data, the [Synthetic Data Vault](https://sdv.dev/) is likely the most applicable to forging your synthetic data. Devised by MIT researchers, the Synthetic Data Vault learns the distribution of tabular and time-series data to create synthetic data that is nearly indistinguishable from your authentic data. This is opposed to random data generators such as [Mockaroo](https://mockaroo.com/) that generate data stochastically such that it doesn't capture the distribution of realistic data. 

When it comes to generating visual data instead of tabular or time-series data, it's a bit harder and requires more domain expertise. To create truly robust synthetic data, you need a 3D model(s) or a 3D scene(s) of the object(s) on which you plan to train a classifier. Next, you have to incorporate your 3D model(s)/scene(s) into a rendering pipeline to actually generate synthetic images. However, this alone is not enough to generate robust synthetic data, and will ultimately lead to overfitting on a select set of 3D models/scenes. You must also introduce [domain randomization](https://arxiv.org/abs/1703.06907) into your synthetic data production pipeline in order to train a model that doesn't overfit your data. To achieve this, techniques such as random camera placement, Gaussian Blurring, Gaussian Noise, random backgrounds, and random colorization can be applied. The list of domain randomization techniques goes on and on, though the techniques previously mentioned are often some of the most effective for creating a dataset that can train a strong computer vision model. We'll use a subset of these techniques when we go ahead and train our own model.

![Hotdog](/article1/sdv.png)
The Synthetic Data Vault production process | Towards Data Science

## Hotdog vs No Hotdog

Now for the exciting part of this blog, where we'll go ahead and build our synthetic dataset, as well as our computer vision model. We'll use [Pytorch](https://pytorch.org/) to take care of the deep-learning related work for our pipeline, and we'll use [Blender](https://www.blender.org/) to take care of the rendering aspect of our pipeline. More specifically, I'll be using a simple script that I created for generating synthetic data in Blender that you can find [here](). This tool can import Blender objects (.dae files), apply a random uniform rotation to the object, apply a random background, Gaussian Noise, and Gaussian Blur, and then render the final image. You can find the 3D .dae files [here](), and the background images that I used [here](). Please note that I opted to use .dae files instead of the more common .obj files due to their ease of integration to the Blender rendering environment. 

Next, we'll create a simple binary classifier in Pytorch to train on our synthetic data. In my case, I created the binary classifier in Google Colab because my GTX 1650 isn't necessarily purpose-built for training computer vision models, and I'm too broke to rent out a Tesla v100 or P100 Cloud GPU.

![Hotdog](/article1/render_00003.png)
An example synthetic hotdog. Notice the granularity caused by Gaussian Noise, as well as the background.

### Creating our Data 
First, we have to create our synthetic data. As mentioned above, we'll go ahead and import our 3D objects as well as our background images. I went ahead and downloaded twenty images to use as backgrounds from Flickr: ten images of tables (you gotta eat somewhere) and ten images of ballparks (when else would a normal person eat hotdogs?). Next, I went ahead and found some 3D hotdogs on [Sketchfab](https://sketchfab.com/search?q=tag%3Ahotdog&sort_by=-relevance&type=models). One was low-poly and one was just a pig in a blanket, but I'm too cheap to go ahead and purchase the professionally created 3D hotdog models. Once I had these, I made sure to convert my 3D objects into the .dae format, and I went ahead and put them in the proper directory for the Synthblend script. Finally, I went ahead and rendered 5,000 images of hotdogs because why not. Leveraging parallel mapping via the [Joblib](https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html) library, I was able to render a hardy 120 images a minute on my Dell XPS 15.

Please note that all 3D objects and images used in making our synthetic dataset have a creative commons license, though I don't plan on creating a startup with my hotdog detecting algorithm anytime soon (that niche has already been taken care of, unfortunately.) You can find the backgrounds and the objects in their respective folders at [this]() repository.

![Hotdog](/article1/render_00200.png)
An example of a synthetic taco. Who's hungry?

### Importing our Data
We'll import the necessary Pytorch packages to create our model. We'll need the base ```torch``` and ```torchvision``` libraries. We also need to load in our data from two separate folders: one containing images of hotdogs, and another containing images of food that isn't a hotdog.
```python
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

from torchvision import datasets, transforms
```
We should also set some default global variables for configuring our model, optimizer, and training loop later on. We can go ahead and define these globals as the following
```python
BATCH_SIZE = 64
BETA_1 = 0.9
BETA_2 = 0.99
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
EPOCHS = 10
LEARNING_RATE = 1e-4
```
Next, we'll go ahead and define our data transforms. We won't get fancy with the augmentations, since the domain randomization techniques applied when the synthetic data was created should suffice. We'll simply crop each image to be 128 by 128 pixels, and then normalize each pixel within the range of -1 and 1.
```python
transform = transforms.Compose([
    transforms.Resize(128),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
```
Finally, we'll go ahead and load in our data using the ```torchvision``` ImageFolder class. We'll also create a training and testing split to our authentic data to record the performance of our model. We won't use a validation set in our data since we won't be performing any hyperparameter tuning, and we'll stochastically split our authentic data as 80% training and 20% testing. We won't split our synthetic data into separate training and testing sets because we don't want our binary classifier to perform well on synthetic data; we intend for our classifier to perform well on authentic data. We'll also go ahead and download our dataset from [Kaggle](https://www.kaggle.com/dansbecker/hot-dog-not-hot-dog), though we won't use it just yet. For now, we'll create a directory structure as follows.
```
📂synthetic
    📂hotdog
    📂not_hotdog
```
Where synthetic is a directory containing the hotdog and not_hotdog subdirectories. These, as the name suggests, contain our synthetic images of hotdogs and not hotdogs. We can finally create our datasets and dataloaders in Pytorch as follows.
```python
synthetic = datasets.ImageFolder(root='./synthetic', transform=transform)
authentic = datasets.ImageFolder(root='./authentic', transform=transform)

authtrain, authtest = random_split(authentic, [799, 199])

synthloader = DataLoader(dataset=synthetic, batch_size=BATCH_SIZE, shuffle = True, drop_last = True)
authloader = DataLoader(dataset=authtrain, batch_size=BATCH_SIZE, shuffle = True, drop_last = True)
testloader = DataLoader(dataset=authtest, batch_size=BATCH_SIZE, shuffle=True)
```

### Creating our Model
For all intents and purposes, we'll use a simple network with residual blocks and average pooling. Could we implement a State-of-the-Art model such as [EfficientNet](https://arxiv.org/abs/1905.11946) with State-of-the-Art activation functions such as the [Funnel Rectified Linear Unit](https://arxiv.org/abs/2007.11824)? Of course, we could! Is that way beyond the scope of an article discussing detecting hotdogs? 100%. Instead, we'll implement a super simple Frankenstein [ResNet](https://arxiv.org/pdf/1512.03385.pdf) architecture without downsampling. We can start by creating our own custom residual block that ties together a 1x1 convolution with a 3x3 padded convolution.

```python
class ResBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c 
        self.relu = nn.ReLU()
        self.layers = nn.Sequential(
            nn.Conv2d(self.in_c, self.out_c, 1),
            nn.BatchNorm2d(self.out_c), 
            self.relu,
            nn.Conv2d(self.out_c, self.out_c, 3, padding=1),
            nn.BatchNorm2d(self.out_c)
        )
        
    def forward(self, x):
        res = x
        x = self.layers(x)
        x = res + x
        return self.relu(x)
```
With that, we can go ahead and create our Binary Classifier! Note that we have a linear output layer that outputs a (BATCH_SIZE x 1) tensor, though this should be amended to (BATCH_SIZE x 2) if you plan to use Cross Entropy Loss instead.
```python 
class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.res32 = ResBlock(32, 32)
        self.res64 = ResBlock(64, 64)
        self.res128 = ResBlock(128, 128)
        self.net = nn.Sequential(
            nn.Conv2d(3,32,7),
            nn.BatchNorm2d(32), # 122
            self.res32,
            self.res32,
            nn.Conv2d(32, 64, 5, stride=2),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            self.res64,
            self.res64,
            nn.Conv2d(64, 128, 5, stride=2),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            self.res128,
            self.res128,
            nn.AvgPool2d(3, 2)
        )
        self.l1 = nn.Linear(512,1)
        self.out = nn.Sigmoid()
        
    def forward(self, x):
        x = self.net(x)
        x = x.reshape(x.shape[0], -1)
        x = self.l1(x)
        x = self.out(x).squeeze()
        return x
```

### Creating the Training Loop
Now that we have all the building blocks in place, it's time that we bring it all together and actually train our model. We'll go ahead and use the ADAM optimizer and train our model for thirty epochs total. We'll also allow for three separate training modes: authentic, synthetic or authentic and synthetic (both). This will allow us to test the accuracy of our model when we train it on all three forms of data later on.
```python
def train(mode: str):  
    if mode not in ['authentic', 'synthetic', 'both']:
        raise ValueError("Mode must be one of \'synthetic\', \'authentic\', or \'both\'")
    optim = Adam(model.parameters(), lr=LEARNING_RATE, betas=(BETA_1, BETA_2))
    criterion = nn.BCELoss()
    
    model.train()
    iters = 0
    
    net_loss = []
    net_accuracy = []
    
    if mode == 'synthetic' or mode == 'both':
        for E in range(EPOCHS):         
            for _, (data, label) in enumerate(synthloader):
                optim.zero_grad()
                data, label = data.to(DEVICE), label.to(DEVICE)

                output = model(data)
                loss = criterion(output.float(), label.float())

                loss.backward()
                optim.step()

                iters += 1

                if iters % 20 == 19:
                    accuracy = torch.sum(torch.round(output) == label)/BATCH_SIZE
                    print(f"iteration: {iters + 1}, loss: {loss.item()}, accuracy: {accuracy.item()}")

        torch.save(model.state_dict(), f'./binary_classifier')    
    
    if mode == 'authentic' or mode == 'both':
        print("\n Fine Tuning \n")

        for E in range((2 * EPOCHS if mode == 'both' else 3 * EPOCHS)):         
            for _, (data, label) in enumerate(authloader):
                model.train()
                optim.zero_grad()

                data, label = data.to(DEVICE), label.to(DEVICE)

                output = model(data)
                loss = criterion(output.float(), label.float())

                loss.backward()
                optim.step()

                iters += 1

                if iters % 20 == 19:
                    # Calculate the accuracy wrt the testing set
                    accuracy = torch.sum(torch.round(output) == label)/BATCH_SIZE
                    print(f"iteration: {iters + 1}, loss: {loss.item()}, accuracy: {accuracy.item()}")
        
        torch.save(model.state_dict(), f"./binary_classifier_{'finetuned' if mode == 'both' else 'authentic'}")
```
Given that we only use four examples of hotdogs and not hotdogs superimposed onto a set amount of preselected backgrounds, our synthetic data is limited in variance. Therefore, training on authentic data serves two purposes: it avoids overfitting on our synthetic data, and it introduces our model to data from the authentic domain (the data we want to actually able to predict on!) We can go ahead and train first on synthetic and authentic, and then solely on authentic as follows 
```python
model = BinaryClassifier().to(DEVICE)
train(mode='authentic')
train(mode='synthetic')
```

## Results 
After training our binary classifier, we can go ahead and test the accuracy of our algorithm. We'll go ahead and create a utility function for calculating the accuracy of our classifier.
```python
def calculate_accuracy(model):
    model.eval()
    
    labels = torch.tensor([])
    outputs = torch.tensor([])
    for _, (data, label) in enumerate(testloader):
        data = data.to(DEVICE)
        output = model(data)
        
        labels = torch.cat((labels, label), dim=0)
        outputs = torch.cat((outputs, output.cpu()), dim=0)
    
    accuracy = (torch.sum(torch.round(outputs) == labels)/len(labels)).item()
    print(f"Model accuracy: {accuracy}")
        
    return accuracy 
```
We can go ahead and run this code using the state dictionaries that we saved for our three separately trained models. The code below will test the accuracy of our model trained solely on authentic data, though you can change the file name to './binary_classifier' and './binary_classifier_both' to determine the model's accuracy on synthetic and synthetic and authentic data, respectively. 

```python
model = BinaryClassifier()
model.load_state_dict(torch.load('./binary_classifier_authentic'))
model.to(DEVICE)
acc = calculate_accuracy(model)
```

After examining our code, we see that our classifier trained solely on synthetic data achieves a dismal 53.77% accuracy on the testing set. This is pretty disappointing for a binary classifier, given that there's a 50% chance of correctly classifying an image stochastically. However, this is where I must emphasize that *synthetic data is not intended to be the sole source of data for training a computer vision model.* Instead, synthetic data is intended to be used in tandem with authentic data so that the computer vision can fine-tune itself. If we instead load in our model's fine-tuned state dictionary, we can see that we achieve a fine 83.92% accuracy on the testing set. This is a pretty nice score given that we didn't implement any hyperparameter searching, minimal data augmentation, and are using a shallow Frankenstein ResNet. When we test our model trained on solely authentic data for three epochs, we find that it achieves an accuracy score of 75.37%. Needless to say, incorporating synthetic data into a production pipeline is a steadfast way to gain a significant performance boost on any algorithm.

If you don't feel as if training for ~60 epochs is enough for either classifier, I urge you to go ahead and modify and the code yourself [here]() to witness the benefits of synthetic data yourself. If you're still not convinced, may I remind you that we superimposed 3D models of hotdogs onto arbitrarily selected background images pulled from SketchFab and Flickr, respectively?

![Hotdog](/article1/138969.jpg)
An example of an authentic hotdog in our dataset.

## Conclusion

Needless to say, synthetic data provides a relatively simple way to increase the accuracy of any computer vision model, especially for domains where collecting data are expensive and/or time-consuming. 

That's not to say that synthetic data is the be-all-end-all for training computer vision models. Creating hyperrealistic renders currently requires a ton of manual tweaking on behalf of a researcher, and generating a single batch of images could take upwards of hours to complete. Furthermore, when 3D objects are superimposed on an image backgorund, any person can immediately pick up the discrepancies present in an image. The lighting on the 3D object may disagree with the lighting of the background image. The object may be way out of proportion to the background (as we saw with the 3D hotdogs superimposed onto backgrounds of tables).

There's still a lot of work to be done in this nascent field, though the initial findings prove promising. With recent advances in graphics rendering, it's not hard to imagine that synthetic data can become a necessity for complex computer vision tasks in the near future. 

