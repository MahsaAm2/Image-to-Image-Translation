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

  ![Simulated Image](https://github.com/user-attachments/assets/2efa0990-f10d-4c6e-b34d-5b4674140379)      ![Simulated Image 2](https://github.com/user-attachments/assets/34c2bfa0-5a7d-4589-961b-f9404cefc73e)       ![Simulated Image 3](https://github.com/user-attachments/assets/ea4ed184-f9e6-4e50-ba0e-296d00d4bb56)


 



- **Domain B**: Real Images

  ![Real Image](https://github.com/user-attachments/assets/72d5758d-72df-4845-bf2f-709e2400f413)           ![Real Image 2](https://github.com/user-attachments/assets/25026c54-271f-4159-81f6-6e0d9ed03097)
  ![Real Image 3](https://github.com/user-attachments/assets/1f6f5f3f-54b3-4beb-865b-e7d5488a1e0f)



