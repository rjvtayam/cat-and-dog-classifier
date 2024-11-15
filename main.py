import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.model_selection import train_test_split

# Cell 1: Import libraries
print("TensorFlow version:", tf.__version__)

# Cell 2: Download data and set key variables
_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)
PATH = path_to_zip.replace('cats_and_dogs.zip', 'cats_and_dogs_filtered')

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')
test_dir = os.path.join(PATH, 'test')

IMG_HEIGHT = 150
IMG_WIDTH = 150
BATCH_SIZE = 32

# Print directory structure
def print_directory_structure(path, level=0):
    print(f"{'  ' * level}{os.path.basename(path)}/")
    if os.path.isdir(path):
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            if os.path.isdir(item_path):
                print_directory_structure(item_path, level + 1)
            else:
                print(f"{'  ' * (level + 1)}{item}")

print("Directory structure:")
print_directory_structure(PATH)

print(f"\nPATH: {PATH}")
print(f"Train directory exists: {os.path.exists(train_dir)}")
print(f"Validation directory exists: {os.path.exists(validation_dir)}")
print(f"Test directory exists: {os.path.exists(test_dir)}")

# Create test set if it doesn't exist
def create_test_set(validation_dir, test_split=0.5):
    cat_files = os.listdir(os.path.join(validation_dir, 'cats'))
    dog_files = os.listdir(os.path.join(validation_dir, 'dogs'))
    
    cat_test, cat_val = train_test_split(cat_files, test_size=test_split, random_state=42)
    dog_test, dog_val = train_test_split(dog_files, test_size=test_split, random_state=42)
    
    # Create test directory
    os.makedirs(test_dir, exist_ok=True)
    
    # Move test files
    for cat in cat_test:
        os.rename(os.path.join(validation_dir, 'cats', cat), os.path.join(test_dir, cat))
    for dog in dog_test:
        os.rename(os.path.join(validation_dir, 'dogs', dog), os.path.join(test_dir, dog))
    
    print(f"Created test set with {len(cat_test) + len(dog_test)} images")

if not os.path.exists(test_dir):
    create_test_set(validation_dir)

# Cell 3: Create image generators
train_image_generator = ImageDataGenerator(rescale=1./255)
validation_image_generator = ImageDataGenerator(rescale=1./255)
test_image_generator = ImageDataGenerator(rescale=1./255)

train_data_gen = train_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                           directory=train_dir,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary')

val_data_gen = validation_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                              directory=validation_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='binary')

test_data_gen = test_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                         directory=test_dir,
                                                         target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                         class_mode=None,
                                                         shuffle=False)

# Cell 4: Plot images
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

sample_training_images, _ = next(train_data_gen)
plotImages(sample_training_images[:5])

# Cell 5: Recreate train_image_generator with augmentation
train_image_generator = ImageDataGenerator(
    rescale=1./255,
    rotation_range=45,
    width_shift_range=.15,
    height_shift_range=.15,
    horizontal_flip=True,
    zoom_range=0.5
)

train_data_gen = train_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                           directory=train_dir,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary')

# Cell 6: Plot augmented images
augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)

# Cell 7: Create and compile the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# Cell 8: Train the model
history = model.fit(
    train_data_gen,
    steps_per_epoch=train_data_gen.samples // BATCH_SIZE,
    epochs=15,
    validation_data=val_data_gen,
    validation_steps=val_data_gen.samples // BATCH_SIZE
)

# Cell 9: Visualize accuracy and loss
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(15)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# Cell 10: Make predictions
predictions = model.predict(test_data_gen)
probabilities = [p[0] for p in predictions]

plt.figure(figsize=(20, 20))
for i, probability in enumerate(probabilities[:50]):
    plt.subplot(10, 5, i+1)
    plt.imshow(plt.imread(os.path.join(test_dir, test_data_gen.filenames[i])))
    plt.title(f"{probability:.2%} Dog")
    plt.axis('off')
plt.tight_layout()
plt.show()

# Cell 11: Check challenge completion
test_loss, test_acc = model.evaluate(val_data_gen)
print(f"Test accuracy: {test_acc:.2%}")
if test_acc >= 0.63:
    print("Congratulations! You've passed the challenge!")
else:
    print("Keep trying to improve your model's accuracy.")