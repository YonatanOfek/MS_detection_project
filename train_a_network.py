import numpy
import matplotlib.pyplot as pyplot
import cv2 as cv
import os
from neural_net import NeuralNet
from data_preprocessing import Data


class TrainingExperiment:
    pass
    def __init__(self, net=NeuralNet(1024), number_of_epochs, size_of_batches, input_vector, label_vector):
        self.NN = net
        pass
    def epoch(self):
        pass
    def batches(self):
        pass
    def plot_metrics(self):
        pass
    


if __name__ == '__main__':

    directory1 = 'C://Users/User/PycharmProjects/MSDetectionProject/MS_Dataset_2019/training'
    data = Data(directory1)
    instances = data.get_instances()

    net1 = NeuralNet(1024)

    m_loss = numpy.zeros(30)
    accuracy = numpy.zeros(30)
    for i in range(30):
        m_loss[i], accuracy[i] = net1.update_batch(instances, 1, len(data.training_pictures))

    pyplot.figure()
    pyplot.plot(accuracy)
    pyplot.figure()
    pyplot.plot(m_loss)
    print(m_loss)
