import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, TimeDistributed, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# Paths to video folders (replace with actual paths)
real_videos_path = '../snap/Celeb-real-20241107T181646Z-001/Celeb-real'
fake_videos_path = '../snap/Celeb-synthesis-20241107T181829Z-001/Celeb-synthesis'

def extract_frames(video_path, max_frames=30, resize=(224, 224)):
    """Extract frames from a video file and resize them."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, resize)  # Resize the frame to the target dimensions
        frames.append(frame)
    cap.release()
    # If not enough frames, pad with zeros
    while len(frames) < max_frames:
        frames.append(np.zeros((resize[0], resize[1], 3)))  # Black frame padding
    return np.array(frames)

def load_data(data_dir, label, max_videos=100):
    """Load video data and return frames with labels."""
    videos, labels = [], []
    for i, filename in enumerate(os.listdir(data_dir)):
        if i == max_videos:
            break
        filepath = os.path.join(data_dir, filename)
        frames = extract_frames(filepath)
        if len(frames) == 30:  # Ensuring all videos have 30 frames
            videos.append(frames)
            labels.append(label)
        else:
            print(f"Skipped {filename}: Not enough frames or corrupted video.")
    return np.array(videos), np.array(labels)

# Load real and fake videos
real_videos, real_labels = load_data(real_videos_path, 0)
fake_videos, fake_labels = load_data(fake_videos_path, 1)

# Combine the real and fake videos data
X = np.concatenate((real_videos, fake_videos), axis=0)
y = np.concatenate((real_labels, fake_labels), axis=0)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model: CNN-LSTM with VGG16
def build_vgg16_lstm_model():
    # Load VGG16 pre-trained on ImageNet, excluding top layers
    vgg16_base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    
    for layer in vgg16_base.layers:
        layer.trainable = False

    model = Sequential()
    model.add(TimeDistributed(vgg16_base, input_shape=(30, 224, 224, 3)))  # Apply VGG16 to each frame
    model.add(TimeDistributed(Flatten()))  # Flatten the output of VGG16
    model.add(LSTM(64, return_sequences=False))  # LSTM to process temporal information
    model.add(Dense(64, activation='relu'))  # Fully connected layer
    model.add(Dropout(0.5))  # Dropout for regularization
    model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification
    return model

# Build and compile the model
model = build_vgg16_lstm_model()
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=200, batch_size=8, validation_data=(X_test, y_test))  # Reduced epochs for quick training

# Plot training and validation accuracy
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='x')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.savefig("200tl.png")

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Confusion Matrix and Classification Report
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)  # Convert probabilities to binary labels

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
