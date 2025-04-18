# Handwritten Digit Recognition with Feedforward Neural Network

> **Disclaimer**: This project is based on the foundational neural network implementation from Michael Nielsen’s open-access book [_Neural Networks and Deep Learning_](http://neuralnetworksanddeeplearning.com/). I built and extended the code to include image preprocessing, custom digit testing, and visualizations.

---

## Network Architecture

| Layer  | Description                                |
|--------|--------------------------------------------|
| Input  | 784 neurons (28×28 grayscale pixels)       |
| Hidden | 30 neurons (sigmoid activation)            |
| Output | 10 neurons (digit 0–9 classification)      |

Training uses stochastic gradient descent (SGD) with backpropagation and mini-batch learning.

---

## File Structure
```
.
├── FNN.py               # Core neural network class
├── mnist_loader.py      # Loads and formats MNIST dataset
├── main.py              # Trains the FNN on MNIST and evaluates accuracy
├── my_own_test.py       # Tests the model using user-provided digit images
└── my_own_test_data/    # Folder with user digit images (0.jpg to 9.jpg)
```

---

## Test on Your Own Handwritten Digits

Place your own digit images (`0.jpg` to `9.jpg`) into the `my_own_test_data/` folder.

Then run:

```bash
python my_own_test.py
```

- Preprocess each image
- Visualize the result
- Predict the digit label
- Print final accuracy (e.g., 7/10)

## Example Output
```
Epoch 0: 9092 / 10000
Epoch 1: 9227 / 10000
...

Image 2.jpg — Predicted: 2, Actual: 2
Image 6.jpg — Predicted: 8, Actual: 6

Custom Test Accuracy: 7/10 (70%)
```
## Results
Achieved ~92%+ accuracy on MNIST test set after 30 epochs

~70% accuracy on personal handwritten images (varies with image quality and style)



