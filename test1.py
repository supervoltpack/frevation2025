import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential, Input, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import gradio as gr

def main():
    # Load histology dataset
    project = "histology"
    prefix = "https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20Towards%20Precision%20Medicine/"
    os.system(f'curl -O "{prefix}images.npy"')
    os.system(f'curl -O "{prefix}labels.npy"')
    images = np.load("images.npy")
    labels = np.load("labels.npy")
    os.remove("images.npy")
    os.remove("labels.npy")

    # Normalize images
    images = images.astype('float32') / 255.0

    # One-hot encode labels
    label_names = np.unique(labels)
    labels_ohe = np.array(pd.get_dummies(labels))

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(images, labels_ohe, test_size=0.2, random_state=42)

    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=15,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    datagen.fit(X_train)

    # Build model using MobileNetV2
    base_model = MobileNetV2(input_shape=X_train.shape[1:], include_top=False, weights='imagenet')
    base_model.trainable = False  # Freeze base

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    predictions = Dense(y_train.shape[1], activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Callbacks
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-6)
    ]

    # Train
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        validation_data=(X_test, y_test),
        epochs=25,
        callbacks=callbacks
    )

    # Evaluate
    loss, acc = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {acc:.4f}")

    # Get true labels and predictions
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)
    print(classification_report(y_true, y_pred, target_names=label_names))

    # ---- Gradio GUI ---- #
    def predict_image(img):
        img = img.resize((X_train.shape[1], X_train.shape[2]))
        img = np.array(img).astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)
        prediction = model.predict(img)[0]
        return {label_names[i]: float(prediction[i]) for i in range(len(label_names))}

    interface = gr.Interface(
        fn=predict_image,
        inputs=gr.Image(type="pil"),
        outputs=gr.Label(num_top_classes=3),
        title="Histology Image Classifier",
        description="Upload a histology image to classify its category using MobileNetV2."
    )

    interface.launch()

if __name__ == "__main__":
    main()
