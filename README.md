# Ex.NO-6-Implementing-a-GAN-Hands-on-project-to-understand-and-implement-a-GAN.
## Aim :To Understand and implement a GAN
Generative Adversarial Network (GAN)
Generative Adversarial Networks (GAN) can generate realistic images by learning from existing image datasets. Here we will be implementing a GAN trained on the CIFAR-10 dataset using PyTorch.
## Procedure:
Step 1: Importing Required Libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvisionfrom torchvision
import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
Step 2: Defining Image Transformations
Use PyTorch’s transforms to convert images to tensors and normalize pixel values between -1 and 1 for better training stability.
transform = transforms.Compose([
transforms.ToTensor(),
transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
Step 3: Loading the CIFAR-10 Dataset
Download and load the CIFAR-10 dataset with defined transformations. Use a DataLoader to process the dataset in mini-batches of size 32 and shuffle the data

train_dataset = datasets.CIFAR10(root='./data',train=True, download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=32, shuffle=True)

Step 4: Defining GAN Hyperparameters
Set important training parameters:
•	latent_dim: Dimensionality of the noise vector.
•	lr: Learning rate of the optimizer.
•	beta1, beta2: Beta parameters for Adam optimizer (e.g 0.5, 0.999)
•	num_epochs: Number of times the entire dataset will be processed (e.g 10)

latent_dim = 100
lr = 0.0002
beta1 = 0.5
beta2 = 0.999
num_epochs = 10

Step 5: Building the Generator
Create a neural network that converts random noise into images. Use transpose convolutional layers, batch normalization and ReLU activations. The final layer uses Tanh activation to scale outputs to the range [-1, 1].
•	nn.Linear(latent_dim, 128 * 8 * 8): Defines a fully connected layer that projects the noise vector into a higher dimensional feature space.
•	nn.Upsample(scale_factor=2): Doubles the spatial resolution of the feature maps by upsampling.
•	nn.Conv2d(128, 128, kernel_size=3, padding=1): Applies a convolutional layer keeping the number of channels the same to refine features.

class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128 * 8 * 8),
            nn.ReLU(),
            nn.Unflatten(1, (128, 8, 8)),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, momentum=0.78),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=0.78),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        return img
Step 6: Building the Discriminator
Create a binary classifier network that distinguishes real from fake images. Use convolutional layers, batch normalization, dropout, LeakyReLU activation and a Sigmoid output layer to give a probability between 0 and 1.
•	nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1): Second convolutional layer increasing channels to 64, downsampling further.
•	nn.BatchNorm2d(256, momentum=0.8): Batch normalization for 256 feature maps with momentum 0.8.
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
        nn.LeakyReLU(0.2),
        nn.Dropout(0.25),
        nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
        nn.ZeroPad2d((0, 1, 0, 1)),
        nn.BatchNorm2d(64, momentum=0.82),
        nn.LeakyReLU(0.25),
        nn.Dropout(0.25),
        nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(128, momentum=0.82),
        nn.LeakyReLU(0.2),
        nn.Dropout(0.25),
        nn.Conv2d(128, 256, kernel_size=3, stride=1,padding=1),
        nn.BatchNorm2d(256, momentum=0.8),
        nn.LeakyReLU(0.25),
        nn.Dropout(0.25),
        nn.Flatten(),
        nn.Linear(256 * 5 * 5, 1),
        nn.Sigmoid()
    )

    def forward(self, img):
        validity = self.model(img)
        return validity
Step 7: Initializing GAN Components
•	Generator and Discriminator are initialized on the available device (GPU or CPU).
•	Binary Cross-Entropy (BCE) Loss is chosen as the loss function.
•	Adam optimizers are defined separately for the generator and discriminator with specified learning rates and betas.
generator = Generator(latent_dim).to(device)discriminator = Discriminator().to(device)
adversarial_loss = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters()
                         , lr=lr, betas=(beta1, beta2))optimizer_D = optim.Adam(discriminator.parameters()
                         , lr=lr, betas=(beta1, beta2))

Step 8: Training the GAN
Train the discriminator on real and fake images, then update the generator to improve its fake image quality. Track losses and visualize generated images after each epoch.
•	valid = torch.ones(real_images.size(0), 1, device=device): Create a tensor of ones representing real labels for the discriminator.
•	fake = torch.zeros(real_images.size(0), 1, device=device): Create a tensor of zeros representing fake labels for the discriminator.
•	z = torch.randn(real_images.size(0), latent_dim, device=device): Generate random noise vectors as input for the generator.
•	g_loss = adversarial_loss(discriminator(gen_images), valid): Calculate generator loss based on the discriminator classifying fake images as real.
•	grid = torchvision.utils.make_grid(generated, nrow=4, normalize=True): Arrange generated images into a grid for display, normalizing pixel values.
for epoch in range(num_epochs):
    for i, batch in enumerate(dataloader):
       
        real_images = batch[0].to(device) 
       
      valid = torch.ones(real_images.size(0), 1, device=device)
      fake = torch.zeros(real_images.size(0), 1, device=device)
       
        real_images = real_images.to(device)

        optimizer_D.zero_grad()
       
z = torch.randn(real_images.size(0), latent_dim, device=device)
      
        fake_images = generator(z)

real_loss = adversarial_loss(discriminator(real_images), valid)
fake_loss=adversarial_loss(discriminator(fake_images.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
    
        d_loss.backward()
        optimizer_D.step()

        optimizer_G.zero_grad()
      
        gen_images = generator(z)
        
        g_loss = adversarial_loss(discriminator(gen_images),valid)
        g_loss.backward()
        optimizer_G.step()
if (i + 1) % 100 == 0:


if (i + 1) % 100 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}]                       Batch {i+1}/{len(dataloader)} "
                f"Discriminator Loss: {d_loss.item():.4f} "
                f"Generator Loss: {g_loss.item():.4f}"
            )
    if (epoch + 1) % 10 == 0:
        with torch.no_grad():
            z = torch.randn(16, latent_dim, device=device)
            generated = generator(z).detach().cpu()
            grid = torchvision.utils.make_grid(generated,
                                        nrow=4, normalize=True)
            plt.imshow(np.transpose(grid, (1, 2, 0)))
            plt.axis("off")
            plt.show()

    


## Conclusion:
Thus successfully implemented and trained a GAN that learns to generate realistic CIFAR-10 images through adversarial training.
