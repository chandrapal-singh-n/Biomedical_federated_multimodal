import os
import torch
import torch.nn as nn
import torchvision
import yaml
from torch.utils.data import DataLoader, Dataset
from dataset_loader import CustomDatasetLoader
from torchvision import transforms, models


# open config 
with open("config_v1.yaml","r") as f:
    config= yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomAffine(
        degrees=10,
        translate=(0.1, 0.1),
        scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


train_root_dir =r"F:\Phd_Documents\federated_multimodal\Brain\Training"

train_dataset= CustomDatasetLoader(train_root_dir,train_transform)
print("Train Dataset Loaded")
# print(train_dataset)
# img, label = train_dataset[0]
# print(img.shape)
# print(label)


test_root_dir =r"F:\Phd_Documents\federated_multimodal\Brain\Testing"

test_dataset = CustomDatasetLoader(test_root_dir,test_transform)
print("Test Dataset Loaded")



# create loader 

train_loader = DataLoader(train_dataset, batch_size=16,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=16,shuffle=False)



# pre-trained 1 vgg

# model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

# model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
# print(model)

# for param in model.features.parameters():
#     param.requires_grad = False


# for param in model.features[24:].parameters(): for vgg 
#     param.requires_grad = True
    

# for param in model.features[5:].parameters():
#     param.requires_grad = True
    

# num_feature = model.classifier[1].in_features


# model.classifier[1]= nn.Linear(num_feature,config['num_class'])  base classifier , now in this iw ill add dropout using sequential 

# model.classifier= nn.Sequential(
#     nn.Dropout(0.5),
#     nn.Linear(num_feature,config['num_class']))  # replacing the classifier structure thats why no model.classifier[1]

# print(model.classifier)


##############
###ResNet

model = models.resnet50(weights =models.ResNet50_Weights.DEFAULT)
print(model)

for param in model.parameters():
    param.requires_grad=False


for param in model.layer3.parameters():
    param.requires_grad=True

for param in model.layer4.parameters():
    param.requires_grad=True
  
num_feature = model.fc.in_features

# model.fc= nn.Linear(num_feature,4)  # replacing the classifier structure thats why no model.classifier[1]

model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_feature, 4)
)

print(model.fc)

model= model.to(device)
print(model)


## loss function and optimizer
criterion =nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(),lr= config['learning_rate'],  weight_decay=float(config['weight_decay']))
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad,
           model.parameters()),
    lr=0.0001,
    weight_decay=1e-4
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.1,
    patience=2
)

#veryfy

for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)
        
    
    
    
#train
best_acc =0
for epoch in range(config['num_epochs']):
    

    model.train()
    running_loss = 0
    correct= 0
    total =0 
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _,predicted= torch.max(outputs,1)
        total +=labels.size(0)
        correct +=(predicted==labels).sum().item()
    train_loss = running_loss/len(train_loader)
    train_acc = 100*correct/total
    
    #now eval 
    model.eval()
    val_loss = 0
    val_correct= 0
    val_total =0
    
    with torch.no_grad():
        
        for images, labels in test_loader:
            images = images.to(device)
            labels= labels.to(device)
            
            # optimizer zero grad no in eval 
            outputs = model(images)
            loss = criterion(outputs,labels)
            
            # no backward no step no update
            val_loss += loss.item()
            _,predicted= torch.max(outputs,1)
            val_total += labels.size(0)
            val_correct += (predicted==labels).sum().item()
    validation_loss = val_loss / len(test_loader)
    val_acc = 100 * val_correct / val_total
    scheduler.step(val_acc)
    current_lr = optimizer.param_groups[0]['lr']
    
    print(
        f"Epoch [{epoch+1}/{config['num_epochs']}] "
        f"LR: {current_lr:.6f} "
        f"Train Loss: {train_loss:.4f} "
        f"Train Acc: {train_acc:.2f}% "
        f"Val Loss: {validation_loss:.4f} "
        f"Val Acc: {val_acc:.2f}%"
    )
    if val_acc>best_acc:
        best_acc=val_acc
        torch.save(
            model.state_dict(), "best_resnet.pth"
        )
        print("best model saved")
