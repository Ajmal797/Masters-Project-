import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, GlobalAveragePooling2D, TimeDistributed, LSTM, Dropout, Input, Attention, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

# Paths to datasets
real_videos_path = '/home/student1/Deepfake_Real'
fake_videos_path = '/home/student1/Deepfake_Sys'

# Function to extract frames from videos
def extract_frames(video_path, max_frames=30, resize=(224, 224)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, resize)
        frames.append(frame)
        frame_count += 1
    cap.release()
    while len(frames) < max_frames:
        frames.append(np.zeros((resize[0], resize[1], 3), dtype=np.uint8))  # Padding
    return np.array(frames)

# Load data
def load_data(data_dir, label, max_videos=100):
    videos, labels = [], []
    for i, filename in enumerate(os.listdir(data_dir)):
        if i == max_videos:
            break
        filepath = os.path.join(data_dir, filename)
        frames = extract_frames(filepath)
        if len(frames) == 30:
            videos.append(frames)
            labels.append(label)
    return np.array(videos), np.array(labels)

# Load real and fake video datasets
real_videos, real_labels = load_data(real_videos_path, label=0)
fake_videos, fake_labels = load_data(fake_videos_path, label=1)

# Combine and shuffle the dataset
X = np.concatenate((real_videos, fake_videos))
y = np.concatenate((real_labels, fake_labels))
X, y = shuffle(X, y, random_state=42)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model with CNN and Attention Mechanism
def build_attention_cnn_model():
    input_layer = Input(shape=(30, 224, 224, 3))  # 30 frames, 224x224 resolution, 3 channels

    # CNN feature extractor for each frame
    cnn = TimeDistributed(Conv2D(32, (3, 3), activation='relu'))(input_layer)
    cnn = TimeDistributed(Conv2D(64, (3, 3), activation='relu'))(cnn)
    cnn = TimeDistributed(GlobalAveragePooling2D())(cnn)  # Pooling to reduce spatial dimensions

    # Apply Attention mechanism
    attention = Attention()([cnn, cnn])  # Self-attention (query and value are the same)

    # Flatten the attention output to feed into LSTM
    attention = TimeDistributed(Flatten())(attention)

    # Apply LSTM to capture temporal dependencies between frames
    x = LSTM(64)(attention)

    # Dense and Dropout layers for classification
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)

    # Output layer for binary classification
    outputs = Dense(1, activation='sigmoid')(x)

    # Create and compile the model
    model = Model(inputs=input_layer, outputs=outputs)
    return model

# Compile the model
model = build_attention_cnn_model()
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=8,
    validation_data=(X_val, y_val)
)

# Evaluate model performance
train_accuracy = history.history['accuracy'][-1]
val_accuracy = history.history['val_accuracy'][-1]
print(f"Final Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"Final Validation Accuracy: {val_accuracy * 100:.2f}%")

# Plotting Accuracy and Loss Graphs
plt.figure(figsize=(12, 5))

# Accuracy Graph
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='x')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()

# Loss Graph
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', marker='o')
plt.plot(history.history['val_loss'], label='Validation Loss', marker='x')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig("attention mech.png")
