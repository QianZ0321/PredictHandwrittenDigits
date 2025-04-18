from PIL import Image, ImageOps
import numpy as np
import FNN
import mnist_loader
import os
import matplotlib.pyplot as plt

def center_digit_image(img, canvas_size=28, digit_box_size=20):
    """
    Centers a digit in a 28x28 image by cropping to its bounding box, 
    resizing to 20x20, and placing it on a black canvas.
    """
    # Get bounding box of the digit
    bbox = img.getbbox()
    if not bbox:
        return Image.new('L', (canvas_size, canvas_size), color=0)

    digit = img.crop(bbox)

    # Resize cropped digit to 20x20
    digit = digit.resize((digit_box_size, digit_box_size), Image.Resampling.LANCZOS)

    # Create new black canvas
    new_img = Image.new('L', (canvas_size, canvas_size), color=0)

    # Center the digit
    upper_left = ((canvas_size - digit_box_size) // 2, (canvas_size - digit_box_size) // 2)
    new_img.paste(digit, upper_left)

    return new_img


def load_custom_image(path):
    """
    Loads, processes, and vectorizes a custom digit image for MNIST-style FNN prediction.
    """
    img = Image.open(path).convert('L')  # Grayscale
    img = ImageOps.invert(img)  # White digit on black background

    # Autocontrast helps bring out the digit better before binarizing
    img = ImageOps.autocontrast(img)

    # Use autocontrasted image to get bbox and center it
    img = center_digit_image(img)

    # Normalize AFTER centering
    img_data = np.asarray(img).astype(np.float32)
    img_data /= 255.0  # Normalize to 0–1

    # Optional thresholding: helps with very faint digits
    # img_data = np.where(img_data > 0.2, 1.0, 0.0)

    img_vector = img_data.flatten().reshape(784, 1)

    # Visualization
    plt.imshow(img_data, cmap='gray')
    plt.title(f"Processed {os.path.basename(path)}")
    plt.show()

    return img_vector


# True labels
true_labels = list(range(10))

# Path to your folder
folder_path = "/Users/x/Desktop/network/my_own_test_data"

# Build test set from handwritten images
my_test_data = []
for i in range(10):
    img_path = os.path.join(folder_path, f"{i}.jpg")
    if not os.path.exists(img_path):
        print(f"[!] Image file not found: {img_path}")
        continue
    img_vector = load_custom_image(img_path)
    my_test_data.append((img_vector, i))  # (image_data, true_label)

# Load MNIST and train
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = FNN.Network([784, 30, 10])
net.SGD(training_data, epochs=30, mini_batch_size=10, eta=3.0)

# Evaluate on custom data with detailed output
correct = 0
for i, (x, y_true) in enumerate(my_test_data):
    y_pred = np.argmax(net.feedforward(x))
    print(f"Image {i}.jpg — Predicted: {y_pred}, Actual: {y_true}")
    if y_pred == y_true:
        correct += 1

# Final accuracy
print(f"\nCustom Test Accuracy: {correct}/10 ({correct * 10}%)")
