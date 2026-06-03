import torch
import torchvision
import torchvision.models as models
import torch.nn as nn

model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
#vgg_feat.eval()
print('VGG feature')
print(model.features)
print('VGG Classifier ')
print(model.classifier)

## freeze feature extraction 

for param in model.features.parameters():
    param.requires_grad=False
    

# replace final layer numer 6 from vgg classifier for 4 class scenario 
num_feature = model.classifier[6].in_features 
print(num_feature)

model.classifier[6]=nn.Linear(num_feature,4)
print(model)


# for name, parm in model.parameters():
#     if param.requires_grad:
#         print(name)

for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)