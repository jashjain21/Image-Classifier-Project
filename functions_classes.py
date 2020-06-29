import torch
from torch import nn
from torchvision import models
from collections import OrderedDict
import numpy as np

def validation(model,validloader,criterion,device):
    '''
    returns validation_loss,accuracy
    '''
    validation_loss=0
    accuracy=0
    for image,label in validloader:
        image,label=image.to(device),label.to(device)
        ps=model.forward(image)
        loss=criterion(ps,label)
        validation_loss+=loss.item()
        log_ps=torch.exp(ps)
        top_k,top_class=log_ps.topk(1,dim=1)
        equal=top_class==label.view(*top_class.shape)
        accuracy+=torch.mean(equal.type(torch.FloatTensor))
    return validation_loss/len(validloader),accuracy


def train(model,criterion, optimizer,trainloader,validloader,epochs=9, print_every=40,device='cpu'):
#resizing would be done
    model.to(device)
    steps = 0
    running_loss = 0
    for e in range(epochs):
        # Model in training mode, dropout is on
        model.train()
        for images, labels in trainloader:
            steps += 1
            images,labels=images.to(device),labels.to(device)
            optimizer.zero_grad()
            #clear the previous gradients
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            #thus  gradients se weights succesfully affected
            running_loss += loss.item()
            #calculated the run time loss
            if steps % print_every == 0:
                # Model in inference mode, dropout is off
                model.eval()
                
                # Turn off gradients for validation, will speed up inference
                with torch.no_grad():
                    validation_loss, accuracy = validation(model, validloader, criterion,device)
                    
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Validation Loss: {:.3f}.. ".format(validation_loss/len(validloader)),
                      "Validation Accuracy: {:.3f}".format(accuracy/len(validloader)))
                
                running_loss = 0
                
                # Make sure dropout and grads are on for training
                model.train()

'''
now the following functions are for predict.py
'''


def load_model(filepath):
    checkpoint = torch.load(filepath)
    model=models.vgg19(pretrained=True)
    for feature in model.parameters():
        feature.requires_grad=False
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 4096)),
                          ('relu1', nn.ReLU()),
                          ('dropout1', nn.Dropout(p=0.2)),
                          ('fc2', nn.Linear(4096, 1000)),
                          ('relu2', nn.ReLU()),
                          ('dropout2', nn.Dropout(p=0.2)),
                          ('fc3', nn.Linear(1000, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    model.classifier=classifier
    #thus recreated our model with the same architecture with the one that we trained with
    model.class_to_idx=checkpoint['class_to_idx'] 
    model.load_state_dict(checkpoint['state_dict'])
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    im=Image.open(image)
    width,height=im.size
    aspect_ratio=width/height
    new_width=256
    new_height=int(new_width/aspect_ratio) 
    newsize=(new_width,new_height)
    im=im.resize(newsize)
    cropped_width,cropped_height=224,224
    left=(new_width-cropped_width)/2
    right=(new_width+cropped_width)/2
    bottom=(new_height+cropped_height)/2
    top=(new_height-cropped_height)/2
    im=im.crop((left,top,right,bottom))
    np_image=np.array(im)
    np_image=(np_image-np_image.mean())/np_image.std()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image=(np_image-mean)/std
    np_image=np_image.transpose((2,1,0))
    return np_image

def predict(image_path, model,device='cpu', topk=5):
    '''
    Predict the class (or classes) of an image using a trained deep learning model.
    '''
    image=process_image(image_path)
    image=torch.from_numpy(image)
    #tensor.unsqueeze_() adds a dimension of size one at the specified position.
    #If you pass the argument dim=0 to the function, it will add a new dimension at position 0
    image.unsqueeze_(0)
    image=image.float()
    model=model.to(device)
    image=image.to(device)
    ps=model.forward(image)
    log_ps=torch.exp(ps)
    probs,classes=log_ps.topk(topk,dim=1)
    return probs.detach().cpu().numpy().flatten(),classes.detach().cpu().numpy().flatten()

def naming(probs,classes,inv_dict,cat_to_name):
    classlist=[]
    for key in classes:
        classlist.append(cat_to_name[inv_dict[key]])
    return classlist