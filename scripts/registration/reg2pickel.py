import pandas as pd
import numpy as np 
import os 
import openslide
import qt_wsi_reg.registration_tree as registration

import pickle as p

parameters = {
    # feature extractor parameters
    "point_extractor": "sift",  #orb , sift
    "maxFeatures": 2048, 
    "crossCheck": False, 
    "flann": False,
    "ratio": 0.7, 
    "use_gray": False,

    # QTree parameter 
    "homography": True,
    "filter_outliner": False,
    "debug": True,
    "target_depth": 1, 
    "run_async": True,
    "num_workers": 8,
    "thumbnail_size": (7000, 7000)
}

files = next(os.walk("/home/heckerm/data/CK/"))[2]

he_files = list()
cy_files = list()

for file in files:
    if "HE" in file and not "thumbnail" in file:
        he_files.append(file)
    elif "CK" in file and not "thumbnail" in file:
        cy_files.append(file)
    else:
        pass



#iterate over both lists 
for he in range(len(he_files)):
    for cy in range(len(cy_files)):
        if he_files[he][:-8] == cy_files[cy][:-8]: #Just the picture name, without the ending "_HE.mrxs" or "_CK.mrxs"
            column_name = he_files[he][:-7]+"CY+HE" #Creating the column name for the csv file 
            #registration
            qtree = registration.RegistrationQuadTree(source_slide_path="/home/heckerm/data/CK/"+str(cy_files[cy]), target_slide_path="/home/heckerm/data/CK/"+str(he_files[he]), **parameters)
            
            print(column_name)
            p.dump(qtree, open(column_name+".p", "wb" ) )