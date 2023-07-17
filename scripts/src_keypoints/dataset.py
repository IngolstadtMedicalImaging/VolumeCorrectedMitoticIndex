import pickle as p 
import pandas as pd
import numpy as np 
import openslide 
import matplotlib.pyplot as plt 

from skimage.color import rgb2hed, hed2rgb
from skimage.exposure import rescale_intensity
from skimage import morphology
import ast 

import cv2 as cv
import torch 

from torch.utils.data import Dataset


class Training_Dataset(Dataset): 

        """ Dataset for epithelial tissue segmentation

    Args:
        img_wid (int): Patch width 
        img_hei (int): Patch height 
        img_cha (int): Image channels (input)
        img_out (int): Image channels (output)
        patch_level (int): Patch downsample level 
        transforms (optional): Augmentations 
        epoch_len (int): Pseudo epoch length
        csv_file_coords (pd.DataFrame): Csv file containing the coordinates of H&E-WSIs and CK-WSIs

    """
    
    def __init__(self, 
        img_wid:int = 512,
        img_hei:int = 512,
        img_cha:int = 3,
        img_out:int = 1,
        patch_level:int = 0,
        transforms = None,
        epoch_len:int = 20,
        file_path:str = "/home/heckerm/data/CK/",
        csv_file_coords=pd.read_csv('/home/heckerm/bachelor/registration/key_coords_train_n3w_reg_0-1_manu.csv', sep=";")):
        
        self.img_wid = img_wid
        self.img_hei = img_hei
        self.img_cha = img_cha
        self.img_out = img_out
        self.patch_level = patch_level
        self.transforms = transforms 
        self.epoch_len = epoch_len
        self.file_path = file_path
        self.csv_file_coords = csv_file_coords
        
        self.objects = self.load_slide_objects()
        self.samples = self.generate_samples()

    def load_slide_objects(self):
        
        df = self.csv_file_coords
        df = df.applymap(ast.literal_eval)
        
        objects = {}
        
        for idx in range(len(self.csv_file_coords.columns)):
            
            slide_names = df.columns[idx]
            
            he_name = slide_names.split("+")[0]
            cy_name = slide_names.split("+")[1]
            

            objects[idx] = (openslide.open_slide(str(self.file_path+he_name+".mrxs")), openslide.open_slide(str(self.file_path+cy_name+".mrxs")), slide_names)
        
        
        return objects 
    
    def generate_patches(self, idx):
        
        df = self.csv_file_coords
        df = df.applymap(ast.literal_eval)
        
        he_slide, cy_slide, slide_names = self.objects[idx]
        
        while True:
            coordinates = df[slide_names][np.random.choice(len(self.csv_file_coords.columns))]
            
            if coordinates is not False:

                if coordinates[1][0] > 500:
                
                    he_coordinate = coordinates[1]      
                    cy_coordinate = coordinates[0]
                    break
        
        return (he_coordinate, cy_coordinate, he_slide, cy_slide)
    
    def generate_samples(self):
        
        samples = {}
        
        for i in range(self.epoch_len):
            
            random_object = np.random.randint(0, len(self.csv_file_coords.columns))
            
            samples[i] = self.generate_patches(random_object)
            
        return samples 
    
    def generate_new_samples(self):
        self.samples = self.generate_samples()
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        couple_sample = self.samples[index]
        
        he_coordinate, cy_coordinate = couple_sample[0], couple_sample[1]
        he_slide, cy_slide = couple_sample[2], couple_sample[3]
        
        he_patch = np.array(he_slide.read_region((he_coordinate), level=self.patch_level, size=(self.img_wid,self.img_hei)).convert('RGB'))
        cy_patch = np.array(cy_slide.read_region((cy_coordinate), level=self.patch_level, size=(self.img_wid,self.img_hei)).convert('RGB'))
        
        ihc_hed = rgb2hed(cy_patch)
        null = np.zeros_like(ihc_hed[:, :, 0])
        d = rescale_intensity(ihc_hed[:, :, 2], out_range=(0,1),
                            in_range=(0, np.percentile(ihc_hed[:, :, 2], 99)))
        zdh = np.dstack((d, null, null))
        
        ret,thresh_mask = cv.threshold(zdh,0.15,1,cv.THRESH_BINARY)
        
        if np.percentile(ihc_hed[:, :, 2], 99) <0.075:
            thresh_mask = np.zeros_like(ihc_hed[:, :, :])
        
        if self.transforms is not None:
            augmentations = self.transforms(image=he_patch, mask=thresh_mask)
            he_patch = augmentations['image']
            thresh_mask = augmentations['mask']
        
        
        he_patch = torch.from_numpy(he_patch).permute(2, 0, 1).type(torch.float32)
        
        bool_mask = (thresh_mask[:,:,0]/255).astype(dtype=bool)
        bool_mask = morphology.remove_small_objects(bool_mask, 2548, connectivity=30)
        bool_mask = morphology.area_closing(bool_mask, 896)        
        bool_mask = np.expand_dims(bool_mask, axis=0)
        bool_mask = torch.from_numpy(bool_mask)
        
        return he_patch, bool_mask


class Validation_Dataset(Dataset): 

      """ Dataset for epithelial tissue segmentation

    Args:
        img_wid (int): Patch width 
        img_hei (int): Patch height 
        img_cha (int): Image channels (input)
        img_out (int): Image channels (output)
        patch_level (int): Patch downsample level 
        transforms (optional): Augmentations 
        epoch_len (int): Pseudo epoch length
        csv_file_paths (pd.DataFrame): Csv file containing the paths of H&E-WSIs, CK-WSIs and qtree

    """

    def __init__(self, 
        img_wid:int = 512,
        img_hei:int = 512,
        img_cha:int = 3,
        img_out:int = 1,
        patch_level:int = 0,
        transforms = None,
        valid_len:int = 100,
        csv_file_paths=pd.read_csv('/home/heckerm/bachelor/registration/path_validation_final.csv', sep=";")):
        
        self.img_wid = img_wid
        self.img_hei = img_hei
        self.img_cha = img_cha
        self.img_out = img_out
        self.patch_level = patch_level
        self.transforms = transforms 
        self.valid_len = valid_len
        self.csv_file_paths = csv_file_paths
        
        self.objects = self.load_slide_and_qtree_objects()
        self.samples = self.generate_samples()

    def load_slide_and_qtree_objects(self):
        
        df = self.csv_file_paths
        objects = {}
        
        for idx in range(len(self.csv_file_paths.index)):
            objects[idx] = (openslide.open_slide(df['he_slide'][idx]), openslide.open_slide(df['cy_slide'][idx]), p.load(open(df['qtree'][idx], 'rb')))
        
        
        return objects 
    
    def generate_patches(self, idx):
        
        he_slide, cy_slide, qtree = self.objects[idx]
        
        slide_width, slide_height = cy_slide.dimensions
        
        while True:
            x_coordinate = np.random.randint(self.img_wid, slide_width-self.img_wid)
            y_coordinate = np.random.randint(self.img_hei, slide_height-self.img_hei)
        
            cy_patch_check = np.array(cy_slide.read_region((x_coordinate, y_coordinate), level=self.patch_level, size=(self.img_wid,self.img_hei)).convert('L'))
            cy_patch_thresh = cv.inRange(cy_patch_check, 80, 180)
            cy_patch_thresh[cy_patch_thresh == 255] = 1
            
            gate = cy_patch_thresh.sum()
            
            if gate > 8000: #25K
                box = [x_coordinate, y_coordinate, 512, 512]
                trans_box = qtree.transform_boxes(np.array([box]))[0]

                if int(trans_box[2].round(0)) == 512:
                    break
        
        box = [x_coordinate, y_coordinate, self.img_wid, self.img_hei]
        trans_box = qtree.transform_boxes(np.array([box]))[0]
        
        cy_coordinate = (x_coordinate, y_coordinate)
        he_coordinate = (int(trans_box[0].round(0)), int(trans_box[1].round(0)))
        
        return (he_coordinate, cy_coordinate, he_slide, cy_slide)
    
    def generate_samples(self):
        
        samples = {}
        
        for i in range(self.valid_len):
            
            random_object = np.random.randint(0, len(self.csv_file_paths.index))
            
            samples[i] = self.generate_patches(random_object)
            
        return samples 
    
    def generate_new_samples(self):
        self.samples = self.generate_samples()
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        couple_sample = self.samples[index]
        
        he_coordinate, cy_coordinate = couple_sample[0], couple_sample[1]
        he_slide, cy_slide = couple_sample[2], couple_sample[3]
        
        he_patch = np.array(he_slide.read_region((he_coordinate), level=self.patch_level, size=(self.img_wid,self.img_hei)).convert('RGB'))
        cy_patch = np.array(cy_slide.read_region((cy_coordinate), level=self.patch_level, size=(self.img_wid,self.img_hei)).convert('RGB'))
        
        ihc_hed = rgb2hed(cy_patch)
        null = np.zeros_like(ihc_hed[:, :, 0])
        d = rescale_intensity(ihc_hed[:, :, 2], out_range=(0,1),
                            in_range=(0, np.percentile(ihc_hed[:, :, 2], 99)))
        zdh = np.dstack((d, null, null))
        
        ret,thresh_mask = cv.threshold(zdh,0.15,1,cv.THRESH_BINARY)
        
        if np.percentile(ihc_hed[:, :, 2], 99) <0.085:
            thresh_mask = np.zeros_like(ihc_hed[:, :, :])
        
        if self.transforms is not None:
            augmentations = self.transforms(image=he_patch, mask=thresh_mask)
            he_patch = augmentations['image']
            thresh_mask = augmentations['mask']
        
        he_patch = torch.from_numpy(he_patch).permute(2, 0, 1).type(torch.float32)
        
        bool_mask = (thresh_mask[:,:,0]/255).astype(dtype=bool)
        bool_mask = morphology.remove_small_objects(bool_mask, 2548, connectivity=30)
        bool_mask = morphology.area_closing(bool_mask, 1296)
        bool_mask = np.expand_dims(bool_mask, axis=0)
        bool_mask = torch.from_numpy(bool_mask)
        
        return he_patch, bool_mask