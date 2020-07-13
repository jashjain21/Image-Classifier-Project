# Image-Classifier-Project
## AI Programming with Python Nanodegree-Udacity

## Project: Create an Image Classifier
### Description
Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smartphone app.
To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture.
A large part of software development in the future will be using these types of models as common parts of applications.
In this project, you'll train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the
flower your camera is looking at. In practice, you'd train this classifier, then export it for use in your application. We'll be using this dataset of 102 flower categories.
When you've completed this project, you'll have an application that can be trained on any set of labelled images. Here your network will be learning about flowers and end up
as a command line application. But, what you do with your new skills depends on your imagination and effort in building a dataset.

### Part 1 - Developing an Image Classifier with Deep Learning
In this first part of the project, I implemened an image classifier with PyTorch.

### Part 2 - Building the command line application
Build a pair of Python scripts that run from the command line to train an image classifier and to predict new images using the trained model.

### Installations
This project requires Python 3.x and the following Python libraries installed:

* Argparse
* Numpy
* Pytorch
* Json
* Pillow
* Seaborn
* Matplotlib

### Files Descriptions
* Image Classifier Project.html:A html version of the project file
* Image Classifier Project.ipynb: A Jupyter notebook contains code to implement an image classifier with PyTorch.
* train.py: Code to train a new deep learning network on a dataset and save the model as a checkpoint.
* predict.py: Code uses a trained network to predict the class for an input image.
* functions_classes.py: Code for functions and classes relating to the model 
* functions_utilities: Code for utility functions like loading data and preprocessing images
* Image Classifier Project.ipnb: Instance of the entire code run on a jupyter notebook
* predict_proof.png: Image depicting the accuracy of the model when run from terminal
* screenshot_training_completion.png: Image depicting the completion of the training over the epochs and the corresponding validation and train losses 
* cat_to_name.json: File having a dictionary of the numbers mapped to the name of flowers (should be placed in the main directory itself)
* workspace-utils.py: File used to continue running the jupyter notebook for a considerably long time

### Data
You can download the data used in this project from [here](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html).
Save the data in a directory named data_dir='flowers'
                                   train_dir = data_dir + '/train'
                                   valid_dir = data_dir + '/valid'
                                   test_dir = data_dir + '/test'

Running codes
To train a new network on a data set with train.py:

Basic usage: python train.py data_directory
### Options:
* Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
* Choose architecture: python train.py data_dir --arch "vgg13"
* Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
* Use GPU for training: python train.py data_dir --gpu
* To predict flower name from an image with predict.py:

Basic usage: python predict.py /path/to/image checkpoint
### Options:
Return top K most likely classes: python predict.py input checkpoint --top_k 3
Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
Use GPU for inference: python predict.py input checkpoint --gpu

Author:
Jash Jain [linkedin](https://www.linkedin.com/in/jash-jain-bb659a132)

License
This project is licensed under the MIT License - see the LICENSE.md file for details.
