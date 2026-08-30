import tensorflow as tf

# Define hyperparameters
BATCH_SIZE = 16  
IMG_SIZE = (224, 224)  
DATA_DIR = 'Dataset_Faces'

print("Loading training data...")
# Load training dataset (80% of the images)
train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    shuffle=True,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

print("\nLoading validation data...")
# Load validation dataset (20% of the images)
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    shuffle=True,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)
print("\nBuilding the Transfer Learning model...")

# Data Augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1)
], name="Data_Augmentation")

# Load the Pre-trained Base Model
IMG_SHAPE = (224, 224, 3)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SHAPE, include_top=False, weights='imagenet'
)
base_model.trainable = False

# Create Custom Classification Head
global_average_layer = tf.keras.layers.GlobalAveragePooling2D(name="Global_Pooling")
prediction_layer = tf.keras.layers.Dense(4, activation='softmax', name="Final_Predictions")

# Build the Final Model Architecture
inputs = tf.keras.Input(shape=IMG_SHAPE)
x = data_augmentation(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)

model = tf.keras.Model(inputs, outputs, name="NBA_Star_Classifier")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# Implement a early stopp mechanism which can decide whether to stop training
early_stop_1 = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',           
    patience=8,                   
    restore_best_weights=True
)

initial_epochs = 30
print("\nStarting the initial training process...")

# First stage training
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=initial_epochs,
    callbacks=[early_stop_1]
)
# Unfreeze top 20 layer for fine tuning       
print("\nUnfreezing the top layers of the base model for Fine-Tuning...")

base_model.trainable = True
fine_tune_at = len(base_model.layers) - 20
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), 
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)
early_stop_2 = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',           
    patience=8,                   
    restore_best_weights=True
)
# Second training
print("\nStarting Fine-Tuning process...")
history_fine = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=20,
    callbacks=[early_stop_2]
)

# Save the training result
model.save('nba_star_classifier.keras')
print("\n Model saved successfully as 'nba_star_classifier.keras'!")
