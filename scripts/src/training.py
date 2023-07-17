import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
import torch.optim as optimizer
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryJaccardIndex
from torch.utils.tensorboard import SummaryWriter

import unet
from dataset import Training_Dataset, Validation_Dataset 
from utils import save_best_model


#Initialize
writer = SummaryWriter("tensorboard_results")

device = torch.device('cuda')
model = unet.build_unet()
model = model.to(device)

iou_all = list()
loss_all = list()

def train(loader, model, optimizer, loss_f, device, epoch):
    model.train()
    
    train_loss = 0
    train_iou = 0
    
    for batch_idx, (data, target) in enumerate(loader):
        
        data = data.to(device, dtype=torch.float32)
        target = target.to(device, dtype=torch.float32)

        
        optimizer.zero_grad()
        output = model(data)
        
        loss = loss_f(output, target)
        loss_value = loss.item()
        train_loss += loss_value

        metric = BinaryJaccardIndex().to(device)
        iou = metric(output, target.type(torch.int32))
        iou_all.append(iou.item())
        train_iou += iou.item()
        
        loss.backward()
        optimizer.step()
        
    train_loss /= len(loader)
    train_iou /= len(loader)
    
    writer.add_scalar("Training loss", train_loss, epoch)
    writer.add_scalar("Intersection over Union - Training", train_iou, epoch)
    
    print(f'\nEpoche: {epoch} with train_loss: {train_loss} and train_iou: {train_iou}')
    
    return train_loss

def valid(loader, model, optimizer, loss_f, device, epoch): 
    valid_loss = 0
    valid_iou = 0
    dice_score = 0
    
    model.eval()
    
    with torch.no_grad():
        
        for batch_idx, (data, target) in enumerate(loader):
            
            data = data.to(device, dtype=torch.float32)
            target = target.to(device, dtype=torch.float32)
            
            output = model(data)
            loss = loss_f(output, target)
            loss_value = loss.item()
            valid_loss += loss_value
            
            metric = BinaryJaccardIndex().to(device)
            iou = metric(output, target.type(torch.int32))
            valid_iou += iou.item()
            
            output_sig = torch.sigmoid(output)
            preds = (output_sig > 0.5).float()
            dice_score += (2 * (preds * target.type(torch.int32)).sum()) / ((preds + target.type(torch.int32)).sum())
            
    valid_loss /= len(loader)
    valid_iou /= len(loader)
    dice_score /= len(loader)
            
    writer.add_scalar("Validation loss", valid_loss, epoch)
    writer.add_scalar("Intersection over Union - Validation", valid_iou, epoch)
    writer.add_scalar("Dice Score - Validation", dice_score, epoch)
    
    print(f'Epoche: {epoch} with valid_loss: {valid_loss} and valid_iou: {valid_iou} and valid_dice_score: {dice_score}\n')

def main(num_epochs=600):
    
    mean = np.load("/home/heckerm/bachelor/u-net/calculated_mean.npy")
    std = np.load("/home/heckerm/bachelor/u-net/calculated_std.npy")
    
    train_transform = A.Compose([
    A.Rotate(p=0.24),
    A.HorizontalFlip(p=0.37),
    A.VerticalFlip(p=0.27),
    A.ColorJitter(hue=0.1, p=0.57), 
    A.Normalize(mean=list(mean), std=list(std)),
        ])
    
    valid_transform = A.Compose([ 
    A.Normalize(mean=list(mean), std=list(std)),])
    
    
    loss = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5) 
    
    old_loss = float("inf")

    
    train_dataset = Training_Dataset(transforms=train_transform, epoch_len=350) 
    valid_dataset = Validation_Dataset(transforms=valid_transform, valid_len=100) 
    
    for epoch in range(num_epochs):
        train_loader = DataLoader(train_dataset, batch_size=4, num_workers=6, shuffle=True)
        current_loss = train(train_loader, model, optimizer, loss, device, epoch)
        train_dataset.generate_new_samples()

        old_loss = save_best_model(model, current_loss, old_loss)
        
        valid_loader = DataLoader(valid_dataset, batch_size=4, num_workers=6, shuffle=False)
        valid(valid_loader, model, optimizer, loss, device, epoch)
        valid_dataset.generate_new_samples()



main()