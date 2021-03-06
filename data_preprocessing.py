import numpy
import cv2 as cv
import os


class Data:

    def __init__(self, directory_string):
        self.labels = []
        self.training_pictures = []
        directory = os.fsencode(directory_string)
        for picture in os.listdir(directory):
            filename = os.fsdecode(picture)
            if filename.endswith(".png") or filename.endswith(".jpg"):
                current_image = cv.imread(os.path.join(directory_string, filename), 0)
                self.training_pictures.append(current_image)
                if filename.startswith("neg"):
                    self.labels.append(0)
                elif filename.startswith("pos"):
                    self.labels.append(1)  # todo: add exceptions

    def get_instances(self, indices=-1):
        if indices.all == -1:
            return zip(self.training_pictures, self.labels)
        else:
            
            batch_pictures = [self.training_pictures[i] for i in
                indices.T]
            batch_labels = [self.labels[i] for i in indices.T]
            return zip(batch_pictures, batch_labels)
    
    def get_middle(self):
        pass