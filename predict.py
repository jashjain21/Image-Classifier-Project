import json
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image

from functions_utilities import get_input_args_predict
from functions_classes import load_model,process_image,predict,naming

in_arg=get_input_args_predict()

model=load_model(in_arg.checkpoint)

probs,classes=predict(in_arg.image_directory,model,in_arg.top_k)

inv_dict = {value: key for key, value in model.class_to_idx.items()} 
device=in_arg.gpu
classlist=naming(probs,classes,inv_dict,device,in_arg.category_names)
for i in range(len(classlist)):
    print(f'Flower name:{classlist[i]}\t Probality:{probs[i]}')
