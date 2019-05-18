import numpy
import matplotlib.pyplot as pyplot
import time
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
                                       learning_rate):
        self.m_loss = numpy.zeros(number_of_epochs)
        self.accuracy = numpy.zeros(number_of_epochs)
        
        for i in range(number_of_epochs-1):
            t = time.time()
            self.m_loss[i], self.accuracy[i] = self.epoch(size_of_batches,
                                                          learning_rate)
            elapsed = time.time() - t
            print(i)
            print(elapsed)
        return self.NN
    
    def epoch(self, size_of_batches, learning_rate):
        batches = self.get_batches(size_of_batches)
        for i in range(batches.shape[1]-1):

            instances = self.training_data.get_instances(batches[:, i])
            mean_loss, accuracy = self.NN.update_batch(instances, learning_rate
                                                       , size_of_batches)
        return mean_loss, accuracy
        
    def get_batches(self, size_of_batches):  # returns matrix whose columns are
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
            size_of_batches, int(size_of_dataset/size_of_batches)))
        return batches
    
    def plot_metrics(self):
        
        pyplot.subplot(2, 1, 1)
        pyplot.plot(self.accuracy)
        pyplot.title('Accuracy vs epochs')
        
        pyplot.subplot(2, 1, 2)
        pyplot.plot(self.m_loss)
        pyplot.title('Loss vs epochs')
        
        pyplot.show()
        
    def validate_network(self):
        pass


if __name__ == '__main__':

    training_session = TrainingSession()
    our_pride_and_joy = training_session.train_network_and_keep_metrics(10,
                                                                        64,
                                                                        0.7)
    # there are
    # 512 instances in the dataset
    training_session.plot_metrics()
