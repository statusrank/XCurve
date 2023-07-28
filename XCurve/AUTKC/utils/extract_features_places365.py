import os

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from tqdm import tqdm

def save_feature(model, data_loader, feature_root):
    os.mkdir(feature_root) if not os.path.exists(feature_root) else None
    feature_list, label_list = list(), list()
    with torch.no_grad():
        for imgs, labels in tqdm(data_loader):
            imgs = imgs.cuda()
            features = model(imgs).cpu()

            feature_list.append(features.cpu().data.detach())
            label_list.append(labels.cpu().data.detach())

    feature_list = torch.cat(feature_list, dim=0)
    print(feature_list.size())
    torch.save(feature_list.data, os.path.join(feature_root, 'features.pth'))
    label_list = torch.cat(label_list, dim=0)
    torch.save(label_list.data, os.path.join(feature_root, 'labels.pth'))


def extract_feature(model, data_root):
    traindir = os.path.join(data_root, 'train')
    valdir = os.path.join(data_root, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]))
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=256, shuffle=False,
        num_workers=8, pin_memory=False)
    save_feature(model, val_loader, 'val_feature_resnet50')

    train_dataset = datasets.ImageFolder(
                        traindir,
                        transforms.Compose([
                            transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            normalize,
                        ]))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=256, shuffle=False,
        num_workers=8, pin_memory=False)
    save_feature(model, train_loader, 'train_feature_resnet50')

model_file = 'resnet50_places365.pth.tar'
model = models.resnet50(num_classes=365)
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
model.load_state_dict(state_dict)

del model.fc
model.fc = lambda x:x
model = model.cuda()
model.eval()

extract_feature(model, 'places365_standard')
