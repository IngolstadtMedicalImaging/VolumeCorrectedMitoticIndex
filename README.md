# Volume corrected mitotic index (M/V-index) on hematoxylin and eosin images
This repository contains the associated programming code for the bachelor thesis "Volumenbasierte Berechnung der Mitoseaktivität von Mammakarzinomen mit Hilfe neuronaler Netze auf Hämatoxylin-Eosin-Bildern". First, the relevant epithelial tissue was segmented on the basis of H&E whole slide images (WSIs). Secondly, the mitoses were determined on the H&E WSI with the help of a mitosis detector and the m/v-index was calculated in the area with the highest mitotic activity. 

## Explorative Results 
A total of 50 tissue sections from dogs with confirmed breast carcinoma were available as a data set. These were stained with H&E staining and cytokeratin staining. In total, 100 WSIs were available. The U-Net architecture was chosen for the segmentation model.

In order to automatically generate segmentation masks for the training of the neural network based on the cytokeratin staining, various methods such as colour deconvolution, thresholding, and morphological filters were applied. The individual predicted 512x512 pixel patches of the model were merged into a whole WSI. Using the H&E WSI, the predicted WSI and a mitosis detector, the m/v-index was calculated. 

## Modelling Results

Here you can add a short summary of the results of your model.

## Usage

Here you can add an explanatation of how to use your project.

## Configuration 

Here you can add information about the configuration of your project.


