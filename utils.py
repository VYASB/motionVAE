import os
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torch

class MotionDataParser:
    def __init__(self, data_dir):
        self.data_dir = data_dir
    
    def load_data(self):
        motions = []
        file_names = os.listdir(self.data_dir)
        file_paths = [os.path.join(self.data_dir, file_name) for file_name in file_names]
        
        for file_path in file_paths:
            data = np.load(file_path)
            motion = data['poses'][0:100][:]  # Replace 'motion' with the appropriate array name in your .npz file
            print(motion.shape)
            motions.append(motion)
        
        return np.array(motions)
    
    def split_data(self, motions, split_ratio=0.8):
        train_motions, test_motions = train_test_split(motions, test_size=1 - split_ratio, random_state=42)
        return train_motions, test_motions

##Saves the training and validation graphical loss plots
def save_loss_plot(train_loss, valid_loss):
    ##loss plots
    plt.figure(figsize = (10, 7))
    plt.plot(train_loss, color ='orange', label='train_loss')
    plt.plot(valid_loss, color ='red', label='valid_loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('D:/ShapeShifter23/motionVAE/outputs/loss.jpg')
    plt.show()

class DataParser:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def get_motion(self):
        motions = []
        file_names = os.listdir(self.data_dir)
        data_dir = [os.path.join(self.data_dir, file_name) for file_name in file_names]
        
        for file_path in data_dir:
            data = np.load(file_path)
            motion = data['poses'][0:200][:]  # Replace 'motion' with the appropriate array name in your .npz file
            motions.append(motion)
        
        return np.array(motions)
