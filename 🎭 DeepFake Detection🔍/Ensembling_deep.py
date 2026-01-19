import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LSTM, TimeDistributed, Bidirectional
from tensorflow.keras.applications import Xception
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
    # Pad with black frames if not enough frames
    while len(frames) < max_frames:
        frames.append(np.zeros((resize[0], resize[1], 3), dtype=np.uint8))
    return np.array(frames)

def load_data(data_dir, label, max_videos=100):
    """Load video data and return frames with labels."""
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

# Load real and fake videos
real_videos, real_labels = load_data(real_videos_path, 0)
fake_videos, fake_labels = load_data(fake_videos_path, 1)

# Combine and split data
X = np.concatenate((real_videos, fake_videos), axis=0)
y = np.concatenate((real_labels, fake_labels), axis=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model 1: CNN-LSTM
def build_cnn_lstm_model():
    model = Sequential()
    model.add(TimeDistributed(Conv2D(32, (3, 3), activation='relu'), input_shape=(30, 224, 224, 3)))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(Conv2D(64, (3, 3), activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(64))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    return model

# Model 2: Xception-based CNN-LSTM
def build_xception_lstm_model():
    base_model = Xception(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    model = Sequential()
    model.add(TimeDistributed(base_model, input_shape=(30, 224, 224, 3)))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(64))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    return model

# Model 3: Bidirectional-LSTM
def build_bidirectional_lstm_model():
    model = Sequential()
    model.add(TimeDistributed(Conv2D(32, (3, 3), activation='relu'), input_shape=(30, 224, 224, 3)))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(Conv2D(64, (3, 3), activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(Flatten()))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    return model

# Build models
models = [build_cnn_lstm_model(), build_xception_lstm_model(), build_bidirectional_lstm_model()]
for model in models:
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train models
histories = []
epochs = 200
for i, model in enumerate(models):
    print(f"Training Model {i + 1}")
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=8, validation_data=(X_test, y_test))
    histories.append(history)

# Plot training and validation accuracy for each model
plt.figure(figsize=(12, 8))
for i, history in enumerate(histories):
    plt.plot(history.history['accuracy'], label=f'Model {i + 1} Train Accuracy', marker='o')
    plt.plot(history.history['val_accuracy'], label=f'Model {i + 1} Validation Accuracy', marker='x')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.savefig("ensembling.png")

# Ensemble prediction
def predict_ensemble(video_path):
    frames = extract_frames(video_path)
    frames = frames[np.newaxis, ...]
    predictions = [model.predict(frames)[0][0] for model in models]
    avg_prediction = np.mean(predictions)
    return "Fake" if avg_prediction > 0.5 else "Real"

# Test the ensemble
uploaded_video_path = '../sample_video.mp4'  # Replace with your video path
result = predict_ensemble(uploaded_video_path)
print(f"The uploaded video is predicted to be: {result}")
