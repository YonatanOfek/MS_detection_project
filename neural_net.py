import numpy as np



def ReLU(x):
    return abs(x)*(x > 0)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class NeuralNet:

    def __index__(self, nn_hdim):
        self.nn_hdim = nn_hdim
        nn_input_dim = 1024
        nn_output_dim = 1
        np.random.seed(0)
        w1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
        b1 = np.zeros((1, nn_hdim))
        w2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
        b2 = np.zeros((1, nn_output_dim))
        self.weights = [w1, w2]
        self.biases = [b1, b2]

    def back_propagtion(self, image, label):

        curr_input = image
        curr_input = np.ravel(curr_input).reshape((1, 1024))

        # Forward propagation
        z1 = curr_input.dot(self.weights[0]) + self.biases[0]
        a1 = ReLU(z1)
        z2 = a1.dot(self.weights[1]) + self.biases[1]
        prob_bp = sigmoid(z2)

        # Back propagation
        delta3 = (prob_bp[0][0] - label) * sigmoid_derivative(z2)
        delta2 = delta3.dot(self.weights[1].T) * relu_derivative(z1)

        dw2 = a1.T.dot(delta3)
        dw1 = curr_input.T.dot(delta2)
        d_nabla_b = [delta2,delta3]
        d_nabla_w = [dw1, dw2]

        return d_nabla_b, d_nabla_w, prob_bp

    def update_batch(self, lr, batch_size):
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        loss = []
        accuracy = []

        for image, label in batch:
            db, dw, prob = self.back_propagtion(image, label)
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, dw)]
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, db)]
            loss.append((label - prob) ** 2 / 2)
            accuracy.append(calculate_accuracy(prob[0][0], label))

        # update rule
        self.weights = [w - (lr / batch_size) * dw for w, dw in zip(self.weights, nabla_w)]
        self.biases = [b - (lr / batch_size) * db for b, db in zip(self.biases, nabla_b)]
        mean_loss = calculate_loss(loss, batch_size)
        mean_accuracy = calculate_accuracy_mean(accuracy, batch_size)
        return mean_loss, mean_accuracy

