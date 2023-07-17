import pandas as pd
import numpy as np 
import os 
import openslide

import registration


he_files = list()
cy_files = list()
con = list()

#Create two lists, each with only HE images or only CY images 
for file in files:
    if "HE" in file and not "thumbnail" in file:
        he_files.append(file)
    elif "CK" in file and not "thumbnail" in file:
        cy_files.append(file)
    else:
        pass


parameters = {
    # feature extractor parameters
    "point_extractor": "sift",  #orb , sift
    "maxFeatures": 2048, 
    "crossCheck": False, 
    "flann": False,
    "ratio": 0.63, 
    "use_gray": False,

    # QTree parameter 
    "homography": True,
    "filter_outliner": False,
    "debug": False,
    "target_depth": 1, 
    "run_async": True,
    "num_workers": 4, 
    "thumbnail_size": (8500, 8500)
}

#Iterate over both lists
for he in range(len(he_files)):
    for cy in range(len(cy_files)):
        if he_files[he][:-8] == cy_files[cy][:-8]: 
            column_name = he_files[he][:-5]+'+'+cy_files[cy][:-5] 
            #registration
            try: 
                qtree = registration.RegistrationQuadTree(source_slide_path="/home/heckerm/data/CK/"+str(cy_files[cy]), target_slide_path="/home/heckerm/data/CK/"+str(he_files[he]), **parameters)
                num_keypoints = qtree.max_points #How many registration points were found  
                print(f'Es werden {num_keypoints} Keypoints geladen')
            
                if num_keypoints > 1: #Sorting out very few 
                    dic = qtree.draw_feature_points(num_sub_pic=num_keypoints)
                    
                    #----------------------------- Visualization -----------------------------#   
                    he_slide = openslide.open_slide(str("/home/heckerm/data/CK/"+str(he_files[he])))
                    ck_slide = openslide.open_slide(str("/home/heckerm/data/CK/"+str(cy_files[cy])))
                    
                    for i in range(num_keypoints):
                        
                        crop_he = he_slide.read_region(location=dic.get(i)[0], level=0, size=(512,512))
                        crop_ck = ck_slide.read_region(location=dic.get(i)[1], level=0, size=(512,512))
                        
                
                    df = pd.DataFrame({column_name:dic})
                    con.append(df)
                else:
                    pass
            except: 
                print('Not worked')
            
        else:
            pass

all_csv = pd.concat(con, axis=1)

all_csv.to_csv(path_or_buf="coordinates.csv", sep=";", index=False, na_rep=False)