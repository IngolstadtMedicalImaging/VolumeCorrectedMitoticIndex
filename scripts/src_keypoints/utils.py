import torch 
from datetime import date 



def save_best_model(model, current_loss, old_loss, checkpoint_filepath="/home/heckerm/bachelor/models/keypoints-method/"):

    if current_loss < old_loss: 
        torch.save(model.state_dict(), checkpoint_filepath + str(date.today()) + ".pth")
        saved = True 

    return current_loss

