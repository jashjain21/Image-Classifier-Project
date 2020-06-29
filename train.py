# # The first file, train.py, will train a new network on a dataset and save the model as a checkpoint. The second file, predict.py, uses a trained network to predict the class for an input image. Feel free to create as many other files as you need. Our suggestion is to create a file just for functions and classes relating to the model and another one for utility functions like loading data and preprocessing images. Make sure to include all files necessary to run train.py and predict.py in your submission.
import torch
from torch import nn,optim
from torchvision import datasets, transforms, models
from functions_utilities import get_input_args_train,data_transformations,image_loader
from functions_classes import train,validation
from collections import OrderedDict
# --data_directory
# --arch
# --learning_rate
# --epochs
# --hidden_units
# --gp
in_arg=get_input_args_train()
data_transforms=data_transformations()
image_datasets,image_loaders=image_loader(data_transforms,in_arg.data_directory)


model=models.vgg19(pretrained=True)
for feature in model.features.parameters():
    feature.requires_grad=False
for classifiers in model.classifier.parameters():
    classifiers.requires_grad=True
    
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 4096)),
                          ('relu1', nn.ReLU()),
                          ('dropout1', nn.Dropout(p=0.2)),
                          ('fc2', nn.Linear(4096, in_arg.hidden_units)),
                          ('relu2', nn.ReLU()),
                          ('dropout2', nn.Dropout(p=0.2)),
                          ('fc3', nn.Linear(in_arg.hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
model.classifier=classifier

device=in_arg.gpu
epochs=in_arg.epochs
criterion=nn.NLLLoss()
learning_rate=in_arg.learning_rate
print_every=40
optimizer=optim.Adam(model.classifier.parameters(),lr=learning_rate)

train(model,criterion, optimizer,image_loaders['trainloaders'],image_loaders['validloaders'],epochs, print_every,device)

model.class_to_idx = image_datasets['training_datasets'].class_to_idx


model.cpu()
checkpoint = {'input_size': 25088,
              'output_size': 102,          
              'state_dict': model.state_dict(),
              'class_to_idx':model.class_to_idx 
             }
#just a dictionanary made wherin i defined an architecture of my model
torch.save(checkpoint, in_arg.save_dir)
