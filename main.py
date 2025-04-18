import mnist_loader as mnist_loader
import FNN

if __name__ == "__main__":
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = FNN.Network([784, 30, 10]) # we can change the hidden layer size
    net.SGD(training_data, 30, 10, 3.0, test_data = test_data) # we can modify the learning rate
    # The transcript shows the number of test images correctly recognized by the neural network after each epoch of training.