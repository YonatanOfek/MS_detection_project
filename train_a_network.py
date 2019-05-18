import numpy
import matplotlib.pyplot as pyplot
import cv2 as cv
import os
from neural_net import NeuralNet
from data_preprocessing import Data

TRAINING_DATASET_DIRECTORY = 'C://Users/User/PycharmProjects/MSDetectionPro' \
                             'ject/MS_Dataset_2019/training'
VALIDATION_DATASET_DIRECTORY = 'C://Users/User/PycharmProjects/MSDetectionP' \
                               'roject/MS_Dataset_2019/validation'


class TrainingSession:

    def __init__(self, untrained_net=NeuralNet(1024),training_data=Data(
                TRAINING_DATASET_DIRECTORY), validation_data=Data(
                VALIDATION_DATASET_DIRECTORY)):
        self.NN = untrained_net
        self.network_backup = untrained_net
        self.training_data = training_data
        self.validation_data = validation_data
        self.m_loss = []
        self.accuracy = []
        
    def train_network_and_keep_metrics(self, number_of_epochs, size_of_batches,
                                       plot_flag=1):
        self.m_loss = numpy.zeros(number_of_epochs)
        self.accuracy = numpy.zeros(number_of_epochs)
        
        for i in range(number_of_epochs-1):
            self.m_loss[i], self.accuracy[i] = self.epoch(size_of_batches)

        return self.NN
    
    def epoch(self, size_of_batches):
        batches = self.get_batches(size_of_batches)
        for i in range(batches.shape[1]-1):
            instances = self.training_data.get_instances(batches[:,i]) #??
            mean_loss, accuracy = self.NN.update_batch(instances,
                                                         1, size_of_batches)
        return mean_loss, accuracy
        
    def get_batches(self,size_of_batches):  # returns matrix whose columns are
                                            # lists of  indices for the batches
                                            # todo: replace assertion with
                                            #  automatic removal of
                                            #  instances (to make batches
                                            #  divide the dataset evenly)
        size_of_dataset = len(self.training_data.training_pictures)
        assert (size_of_dataset/size_of_batches).is_integer(), 'The batch ' \
                                                              'size does not'\
                                                              'divide the ' \
                                                              'dataset evenly'
        numpy.random.seed(0)
        batches = numpy.random.permutation(size_of_dataset)
        batches = numpy.ravel(batches).reshape((
            size_of_batches, size_of_dataset/size_of_batches))
        return batches
    
    def plot_metrics(self):
        
        pyplot.figure()
        pyplot.plot(accuracy)
        pyplot.figure()
        pyplot.plot(m_loss)
        print(m_loss)
        pass


if __name__ == '__main__':

    training_session = TrainingSession()
    our_pride_and_joy = training_session.train_network(30, 64) # there are
    # 512 instances in the dataset
    
