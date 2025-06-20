#!/usr/bin/env python
"""
defect_detector.py

Train or inference a CNN model to classify product images as 'OK' or 'Defect'.
Usage:
    python defect_detector.py --mode train --data_dir data/ --epochs 10
    python defect_detector.py --mode infer --image_path sample.jpg
"""
import os
import argparse
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

IMG_SIZE = (128, 128)
MODEL_PATH = "model/defect_cnn.h5"

def build_cnn():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(*IMG_SIZE, 3)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train(data_dir, epochs=10, batch_size=32):
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    datagen = ImageDataGenerator(rescale=1./255,
                                 rotation_range=15,
                                 width_shift_range=0.1,
                                 height_shift_range=0.1,
                                 horizontal_flip=True)

    train_gen = datagen.flow_from_directory(train_dir,
                                            target_size=IMG_SIZE,
                                            batch_size=batch_size,
                                            class_mode='binary')

    val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(val_dir,
                                                                     target_size=IMG_SIZE,
                                                                     batch_size=batch_size,
                                                                     class_mode='binary')

    model = build_cnn()
    model.summary()

    os.makedirs("model", exist_ok=True)
    ckpt = ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
    es = EarlyStopping(monitor='val_accuracy', patience=3, verbose=1, restore_best_weights=True)

    history = model.fit(train_gen,
                        validation_data=val_gen,
                        epochs=epochs,
                        callbacks=[ckpt, es])

    # Plot history
    plt.figure(figsize=(8,4))
    plt.plot(history.history['accuracy'], label='train_acc')
    plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.legend()
    plt.title('Training History')
    plt.savefig("model/training_history.png")
    print("Model saved to", MODEL_PATH)

def grad_cam(model, img_array, layer_name=None):
    import tensorflow as tf
    from tensorflow.keras import backend as K
    # Convert to batch of 1
    img = np.expand_dims(img_array, axis=0)
    preds = model.predict(img, verbose=0)[0]
    pred_index = 0
    class_output = model.output[:, pred_index]
    if layer_name is None:
        layer_name = [l.name for l in model.layers if 'conv' in l.name][-1]
    last_conv_layer = model.get_layer(layer_name)
    grads = K.gradients(class_output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([img])
    for i in range(pooled_grads_value.shape[-1]):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1
    heatmap = cv2.resize(heatmap, IMG_SIZE)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return heatmap

def inference(image_path):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found. Train first.")
    model = load_model(MODEL_PATH)
    img = cv2.imread(image_path)
    img_res = cv2.resize(img, IMG_SIZE)
    img_res_norm = img_res / 255.0
    prediction = model.predict(np.expand_dims(img_res_norm, axis=0))[0][0]
    label = "Defect" if prediction > 0.5 else "OK"
    heatmap = grad_cam(model, img_res_norm)
    # Superimpose heatmap onto image
    overlay = cv2.addWeighted(img_res, 0.6, heatmap, 0.4, 0)
    cv2.imwrite("gradcam_result.jpg", overlay)
    print("Prediction:", label, "(", prediction, ")")
    print("Grad-CAM result saved to gradcam_result.jpg")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Product Defect Detection")
    parser.add_argument("--mode", choices=["train", "infer"], required=True)
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--image_path", help="Path to image for inference")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    if args.mode == "train":
        train(data_dir=args.data_dir, epochs=args.epochs)
    else:
        if args.image_path is None:
            raise ValueError("Please provide --image_path for inference mode")
        inference(args.image_path)
