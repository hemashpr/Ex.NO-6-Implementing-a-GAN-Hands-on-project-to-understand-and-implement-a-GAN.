# Ex. No. 6 - Implementing a Generative Adversarial Network (GAN)

## Aim
To implement and train a Generative Adversarial Network using PyTorch to generate CIFAR-10-like images.

## Software Requirements
- Python 3.x
- PyTorch
- Torchvision
- NumPy
- Matplotlib

Install dependencies using `pip install -r requirements.txt`.

## Dataset
CIFAR-10 contains 50,000 colour training images across 10 classes. Each image is 32 × 32 pixels with 3 colour channels. Torchvision downloads the dataset automatically.

## Algorithm
1. Import PyTorch, Torchvision, NumPy and Matplotlib.
2. Set random seeds and select CUDA or CPU.
3. Load CIFAR-10 and normalize image values to [-1, 1].
4. Create a shuffled DataLoader with batch size 128.
5. Set latent dimension, learning rate, Adam parameters and epoch count.
6. Build the Generator to convert random noise into 32 × 32 colour images.
7. Build the Discriminator to classify images as real or generated.
8. Initialize Binary Cross-Entropy loss and separate Adam optimizers.
9. Train the Discriminator using real images and detached generated images.
10. Train the Generator to make generated images classify as real.
11. Record and print both losses after each epoch.
12. Save generated image samples and plot the loss curves.
13. Save the trained Generator and Discriminator weights.
14. Stop.

## Program
Complete PyTorch source code: [`gan_cifar10.py`](gan_cifar10.py).

## Output
```text
DEVICE: cuda or cpu
TRAINING IMAGES: 50000
IMAGE SIZE: 3 x 32 x 32
BATCH SIZE: 128

Epoch [1/10] Discriminator Loss: (runtime output) Generator Loss: (runtime output)
...
Epoch [10/10] Discriminator Loss: (runtime output) Generator Loss: (runtime output)

GENERATED FILES
gan_training_losses.png
generated_images.png
generated_epoch_01.png to generated_epoch_10.png
generator.pth
discriminator.pth
```

![Generator and discriminator training losses](gan_training_losses.png)

![Generated CIFAR-10 images](generated_images.png)

## Result
A PyTorch GAN was implemented and trained on CIFAR-10 to generate synthetic colour images.
