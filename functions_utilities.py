import argparse
import torch
from torchvision import datasets, transforms, models
import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
data_dir = 'flowers'
def get_input_args_train():
    # Create Parse using ArgumentParser
    parser=argparse.ArgumentParser()
    # Create 3 command line arguments as mentioned above using add_argument() from ArguementParser method
    parser.add_argument('--data_directory',type=str,default=data_dir,help='path to the folder of train and validation data')
    parser.add_argument('--save_dir',type=str,default='checkpoint.pth',help='path to the folder of saving the checkpoint')
    parser.add_argument('--learning_rate',type=int,default=0.001,help='rate at which the model is training')
    parser.add_argument('--epochs',type=int,default=9,help='epochs at which the model is training')
    parser.add_argument('--hidden_units',type=int,default=1000,help='hidden units')
    parser.add_argument('--gpu',type=str,default='cuda',help='mode of traning')
    # Replace None with parser.parse_args() parsed argument collection that you created with this function 
    return parser.parse_args()
def get_input_args_predict():
    parser=argparse.ArgumentParser()
    parser.add_argument('--gpu',type=str,default='cuda',help='mode of traning')
    parser.add_argument('--checkpoint',type=str,default='checkpoint.pth',help='path to the folder of loading the checkpoint')
    parser.add_argument('--top_k',type=int,default=5,help='path to the folder of saving the checkpoint')                    
    parser.add_argument('--category_names',type=str,default='cat_to_name',help='path to the folder of saving the checkpoint')     
    parser.add_argument('--image_directory',type=str,default='flowers/test/1/image_06743.jpg',help='path to the location of image')
    return parser.parse_args()
                        
def data_transformations():
    data_transforms={
        'training_transforms' : transforms.Compose([
                                            transforms.RandomRotation(30),
                                            transforms.RandomResizedCrop(224),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ]),
        'validation_transforms':transforms.Compose([
                                            transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])}
    return data_transforms
# TODO: Load the datasets with ImageFolder
def image_loader(data_transforms,direc='flowers'):
    train_dir = direc + '/train'
    valid_dir = direc + '/valid'
    image_datasets={
    'training_datasets' : datasets.ImageFolder(train_dir,transform=data_transforms['training_transforms']),
    'validation_datasets' : datasets.ImageFolder(valid_dir,transform=data_transforms['validation_transforms'])
    }
    # TODO: Using the image datasets and the trainforms, define the dataloaders
    image_loaders={
    'trainloaders' : torch.utils.data.DataLoader(image_datasets['training_datasets'],batch_size=64,shuffle=True),
    'validloaders' : torch.utils.data.DataLoader(image_datasets['validation_datasets'],batch_size=64)
    }
    return image_datasets,image_loaders




