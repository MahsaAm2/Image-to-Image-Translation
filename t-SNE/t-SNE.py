import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Function to load and preprocess images
def load_images_from_folder(folder):
    images = []
    i = 0
    for filename in os.listdir(folder):
        
        if i == 476:
            break
        i = i + 1
        img = image.load_img(os.path.join(folder, filename), target_size=(224, 224))
        img = image.img_to_array(img)
        img = tf.keras.applications.resnet.preprocess_input(img)  # ResNet preprocessing
        images.append(img)
       
    return np.array(images)

# Load real and simulated images
real_images = load_images_from_folder('./dataset_sim2real_final/testB')
simulated_images = load_images_from_folder('./dataset_sim2real_final/testA')
generated_images = load_images_from_folder('./fake_B')

# Extract features using a pre-trained ResNet model
base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False)
real_features = base_model.predict(real_images)
simulated_features = base_model.predict(simulated_images)
generated_features = base_model.predict(generated_images)

# Combine features and labels
features0 = np.vstack([real_features])
labels0 = np.hstack([np.ones(len(real_features))])

features2 = np.vstack([simulated_features])
labels2 = np.hstack([np.zeros(len(simulated_features))])

features1 = np.vstack([generated_features])
labels1 = np.hstack([np.zeros(len(generated_features))])

# Reshape features array
features_flat = features0.reshape(features0.shape[0], -1)
features_flat1 = features1.reshape(features1.shape[0], -1)
features_flat2 = features2.reshape(features2.shape[0], -1)

# Run t-SNE
tsne = TSNE(n_components=2, random_state=42)
embedded = tsne.fit_transform(features_flat)
tsne1 = TSNE(n_components=2, random_state=42)
embedded1 = tsne1.fit_transform(features_flat1)

tsne2 = TSNE(n_components=2, random_state=42)
embedded2 = tsne2.fit_transform(features_flat2)
# Visualize the embedding

plt.plot(embedded[:,0], embedded[:,1], 'r*', label='real images') 
plt.plot(embedded1[:,0], embedded1[:,1], 'b*', label='generated images')  

plt.xlabel('X data')
plt.ylabel('Y data')
plt.title('real images and generated images')
plt.legend()

plt.show()


