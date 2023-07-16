import numpy as np
import pandas as pd
import openslide 
import matplotlib.pyplot as plt 
import cv2 as cv 
import torch 
from PIL import Image 
from tqdm import tqdm

import unet

import pyvips 
from tma_utils import extract_core, core_2_vips
from skimage.filters import window

import threading
from threading import Event

import sys 
import argparse
import os 

def preparate_patch(patch):
    x = (patch-mean*255) / (std*255)
    x = torch.from_numpy(x).permute(2, 0, 1).type(torch.float32)
    
    return x 

def calculate_split_cords(num_worker, overlap):

    thresh = int(int((slide_size[0]/overlap)-1)/num_worker)

    df = pd.DataFrame(columns=["x", "y"])

    x = 0
    y = 0 

    idx = 0
    all_df = {}

    for x_x in range(int(slide_size[0]/overlap)-1):
        for y_y in range(int((slide_size[1])/overlap)-1): 

            new_row = ({'x':x, 'y':y})
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            y += overlap
            
        x += overlap
        y = 0 
        
        if x_x == thresh:   
            all_df[idx] = df
            idx += 1
            df = pd.DataFrame(columns=["x", "y"])
    all_df[idx] = df

    return all_df

def worker(df, wsi, num_ids, patch_size, level):
    
    for i in range(0, num_ids):
        x = df['x'][i]
        y = df['y'][i]
        
        act_patch = np.array(he_slide.read_region((x, y), level=level, size=(patch_size,patch_size)).convert("RGB"))

        patch = preparate_patch(act_patch)
        patch = patch.to(device)
        #Macht aus True/False -> 1,0

        prediction_model = model(patch[None,:,:,:])
        pred_sigmoid = torch.sigmoid(prediction_model)
        #pred = pred_sigmoid > 0.5

        pred = pred_sigmoid[0].detach().cpu().numpy() #Aus Tensor wird array
        pred = np.squeeze(pred, axis=0) #Dimensionen squezzen
        #pred = pred.astype(int)
        
        pred = pred * window('hann', (patch_size,patch_size)) #pred * 1/4
        #
        try: 
            wsi[y:y+patch_size, x:x+patch_size] += pred 
        except:
            pass  

parser = argparse.ArgumentParser(description='create predicted WSI')
parser.add_argument('--source', type=str, help='path to the H&E-WSI to be processed')
parser.add_argument('--overlap', type=int, default=256, help='patch overlap')
parser.add_argument('--patch_size', type=int, default=512, help='size of the patches')
parser.add_argument('--level', type=int, default=0, help='patch level')
parser.add_argument('--threshold', type=float, default=0.5, help='model threshold for segmentation of the tissue')
parser.add_argument('--save_dir', type=str, help='directory to save the predicted WSI')

device = 'cuda'
model = unet.build_unet()
model = model.to(device)
model.load_state_dict(torch.load("/home/heckerm/bachelor/models/slide-method/2023-05-18.pth", map_location=device))
model.eval()



if __name__ == '__main__':

    args = parser.parse_args()

    wsi_save_dir = os.path.join(args.save_dir, 'predicted_wsi')
    source = args.source
    save_dir = args.save_dir

    save_file = source.split('/')[-1].split('.')[0]

    print('source:', source)
    print('target:', save_dir+save_file+'_ML.tif')

    he_slide = openslide.open_slide(source)
    slide_size = he_slide.level_dimensions[0] 
    
    
    wsi_height = slide_size[1]
    wsi_width = slide_size[0]
    wsi = np.zeros((wsi_height, wsi_width))

    wsi = wsi.astype(np.float16)

    overlap = args.overlap
    patch_size = args.patch_size
    level = args.level
    threshold = args.threshold
    num_worker = 2
    mean = np.load("/home/heckerm/bachelor/u-net/calculated_mean.npy")
    std = np.load("/home/heckerm/bachelor/u-net/calculated_std.npy")


    coords_dic = calculate_split_cords(num_worker, overlap)

    df_1 = coords_dic[0]
    df_2 = coords_dic[1]

    t1 = threading.Thread(target=worker, args=(df_1, wsi, len(df_1.index), patch_size, level))
    t2 = threading.Thread(target=worker, args=(df_2, wsi, len(df_2.index), patch_size, level))

    t1.start()
    t2.start()

    t1.join()
    t2.join()

    wsi = wsi > threshold 
    wsi = wsi.astype(np.uint8) 
    wsi[wsi == 1] = 255
    wsi = wsi.astype(np.uint8)

    vi = pyvips.Image.new_from_memory(wsi, wsi.shape[1], wsi.shape[0], 1, format='uchar')

    vi.tiffsave(save_dir+save_file+'_ML.tif',
            compression='deflate', 
            tile=True, 
            tile_width=128,   # vips default size
            tile_height=128, 
            pyramid=True, 
            bigtiff=True, 
            properties=True)