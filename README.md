# Deep Neural Networks for Image-to-Image Translation

## Project Goal

The primary goal of this project is to achieve unsupervised image translation across different visual domains. To accomplish this, we leverage methods like diffusion models and CycleGAN. Additionally, we address challenges such as mode collapse, convergence issues, and slow learning. This project explores these challenges in detail and proposes solutions to overcome them.

## Dataset

This project uses an unpaired dataset comprising two distinct domains: simulated machine images and real machine images. The goal of converting simulated images to realistic ones has valuable applications in fields like automobile manufacturing and AI robotics. Generating real images for training models can be costly and time-intensive, whereas simulated images are often cheaper and faster to produce. Leveraging simulated images and translating them into realistic ones can therefore reduce the costs and time required to train models effectively.

- **Domain A**: 6,320 simulated images
- **Domain B**: 6,690 real images of cars
- **Image size**: 128 x 128 pixels

## Dataset Examples

Here are sample images from each domain:


- **Domain A**: Simulated Images

<div style="display: flex; justify-content: space-around;">
    <img src="https://github.com/user-attachments/assets/2efa0990-f10d-4c6e-b34d-5b4674140379" style="height: 200px; width: 200px;" />
    <img src="https://github.com/user-attachments/assets/34c2bfa0-5a7d-4589-961b-f9404cefc73e" style="height: 200px; width: 200px;" />
</div>

- **Domain B**: Real Images

<div style="display: flex; justify-content: space-around;">
    <img src="https://github.com/user-attachments/assets/72d5758d-72df-4845-bf2f-709e2400f413" style="height: 200px; width: 200px;" />
    <img src="https://github.com/user-attachments/assets/25026c54-271f-4159-81f6-6e0d9ed03097" style="height: 200px; width: 200px;" />
</div>


## CycleGAN

The **CycleGAN** architecture is designed to learn mappings between two different domains without paired examples. It consists of two generators and two discriminators. The generators learn to transform images from one domain to the other, while the discriminators differentiate between real and generated images in their respective domains. The cycle consistency loss ensures that an image translated to the target domain can be translated back to the original domain, maintaining the essential characteristics of the input image.

### Generator

The **Generator** is designed to perform image translation between different visual domains. It consists of:

- **Feature Map Block**: Initiates the transformation process by mapping input images to a hidden representation.
- **Contracting Blocks**: Two layers that progressively downsample the feature maps while increasing the depth.
- **Residual Blocks**: Nine layers that learn complex features while preserving information from the input.
- **Expanding Blocks**: Two layers that upsample the feature maps back to the original dimensions.
- **Output Feature Map Block**: Produces the final output image in the target domain.

### Discriminator

The **Discriminator** class is designed similarly to the contracting path of a U-Net. Its primary function is to classify input images as real or fake, providing essential feedback to the generators during training. Key components include:

- **Feature Map Block**: Initializes the network by mapping input images to a hidden representation.
- **Contracting Blocks**: Three layers that downsample feature maps while increasing depth, applying convolution operations followed by Leaky ReLU activations.

Unlike traditional discriminators that output a single scalar value, this Discriminator generates a probability map, predicting whether each N × N patch of the image is real or fake. This approach, known as **PatchGAN**, offers several advantages:

1. **Improved Quality**: Produces more accurate textures and styles by focusing on local patches.
2. **Efficient Calculation**: Reduces computational requirements, enhancing efficiency with large datasets.
3. **Robustness**: Demonstrates increased resilience against distortions by concentrating on small local areas.


### Key Components

- **Generators**: 
  - `gen_AB`: Transforms images from domain A to domain B.
  - `gen_BA`: Transforms images from domain B to domain A.

- **Discriminators**:
  - `disc_A`: Discriminates between real images from domain A and fake images generated by `gen_BA`.
  - `disc_B`: Discriminates between real images from domain B and fake images generated by `gen_AB`.

### Loss Functions

The model employs several loss functions to ensure quality image translation:

1. **Discriminator Loss**: Evaluates how well the discriminators can distinguish real from fake images.
2. **Adversarial Loss**: Encourages the generators to produce realistic images that can fool the discriminators.
3. **Identity Loss**: Ensures that when an image from one domain is passed through the generator for its own domain, it remains unchanged.
4. **Cycle Consistency Loss**: Enforces that an image translated from one domain to another and back should return to its original form, promoting coherence between the two domains.

### Training Setup

- **Hyperparameters**:
  - Number of epochs: `n_epochs = 10`
  - Learning rate: `lr = 0.0002`
  - Batch size: `batch_size = 1`
  - Input dimensions: `load_shape = 286`, `target_shape = 256`

### Results

Below are the results of image transfer from Domain A to Domain B using the CycleGAN model.


#### Example Images from Domain A to Domain B

<div style="display: flex; justify-content: space-around;">
    <img src="https://github.com/user-attachments/assets/404f6959-8215-4316-b0da-5c1c59a362be" width="250" />
    <img src="https://github.com/user-attachments/assets/9c1dd6f9-daa2-47be-b0f3-c3227c0d5eb1" width="250" />
</div>

<div style="display: flex; justify-content: space-around;">
    <img src="https://github.com/user-attachments/assets/1e61dad3-658e-4dfc-9dc2-36f32a57cf25" width="250" />
</div>

#### Example Images from Domain B to Domain A

<div style="display: flex; justify-content: space-around;">
    <img src="https://github.com/user-attachments/assets/d8598d3a-e36c-4dca-80c3-1da9bd102ca3" width="250" />
    <img src="https://github.com/user-attachments/assets/b03dfae4-9738-44e2-816d-6246bcc3ae01" width="250" />
</div>

<div style="display: flex; justify-content: space-around;">
    <img src="https://github.com/user-attachments/assets/300e9b06-3734-43b7-94b6-3fabf4bdfe34" width="250" />
</div>

## Diffusion Model

Initial methods for unpaired data, such as **CycleGAN**, aimed to maintain consistency between source and output images using one-sided learning strategies. While effective, they required additional training, leading to inefficiencies.


### Unpaired Schrödinger Bridge (UNSB)

The **Unpaired Schrödinger Bridge (UNSB)** presents an effective solution for unpaired image-to-image translation. UNSB formulates models as adversarial learning problems, using advanced discriminators and regularization techniques.

UNSB effectively manages the complexities of unpaired data and high-resolution images, demonstrating success in various practical applications. Its main objective is to minimize the difference between the actual target distribution and the model distribution, formulated as a Lagrangian optimization problem with Kullback-Leibler divergence constraints. This approach improves dimensional challenges and enhances high-dimensional data management.

### DDGAN

**Denoising Diffusion Generative Adversarial Network (DDGAN)**, derived from UNSB, combines diffusion processes with GANs to generate high-quality images. By gradually adding noise to the original image, DDGAN creates intermediate images that align with the data distribution. This method focuses on noise removal and can translate between arbitrary distributions.

The UNSB equation is based on a Markov chain decomposition, learning transition probabilities between successive time steps, which supports inductive learning and enhances the approach's efficacy.


### Results from Domain A to Domain B

Below are the results of image transfer from Domain A to Domain B:

<div style="display: flex; justify-content: space-around;">
    <img src="https://github.com/user-attachments/assets/a841f97e-5d19-4d6e-b63c-1d1ae1bab010" width="250" height="125"/>
    <img src="https://github.com/user-attachments/assets/5d8dbb47-8b2d-4f8d-b040-6f13884c18e8" width="250" height="125"/>
</div>

<div style="display: flex; justify-content: space-around;">
    <img src="https://github.com/user-attachments/assets/92fc10ae-76af-425b-8223-baef86370756" width="250" height="125"/>
</div>


## Final Results

The image below shows the result of TSNE execution on real data and simulated data. Images of real cars are marked with a red star, while simulated cars are represented with a blue star.
![image](https://github.com/user-attachments/assets/3f4bc34c-bc8c-4a75-8b0b-3788a89c63a6)

In the figure above, we can see the result of TSNE execution on the real data and the data produced by CycleGAN. The overlap observed indicates that the simulated data is effectively mapped to the real data.
![image](https://github.com/user-attachments/assets/abae2d54-e094-40a8-9ee2-f829f22ab3fe)

In the bottom image, we used the model based on diffusion, which demonstrates more overlap than the first image, indicating that the model has successfully performed the mapping.
![image](https://github.com/user-attachments/assets/146fa511-e799-471a-82ca-3c30968ff2d1)











