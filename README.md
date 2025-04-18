This project aims to create a feedforward Neural Network (FNN) that recognizes handwritten digits.

The structure of FNN:
    1. Forward Propogation
        1. Input layer: takes a 28*28 grayscale image of handwritten digits and flattens it into 784*1 vector
        2. Output layer: z = w * x + b, then passed through a sigmoid function.

    2. Loss Calculation

    3. Backpropagation: compute gradients
        - The network calculates how much each weight contributed to the error using the chain rule.
        - It computes gradients of the loss with respect to each weight in the network.

    4. Training: Stochastic Gradient Dscent (SGD)
        For each epoch, the whole MNIST dataset is shuffled and split into mini-batches of specific size.
        The model will update its weights once per mini-batch. And after updating for all batches, one epoch is completed.

            Here is the training loop:
            for epoch in range(epochs):
                shuffle training data
                split into mini-batches
                for each mini-batch:
                    update weights/biases using backpropagation

            After each epoch, it optionally prints how many test inputs were classified correctly.