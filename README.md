
# Sports Bike Image Classifier with Flask API

This project is a deep learning-based image classification system that identifies different types of sports bikes. It consists of two main components:
1. A Convolutional Neural Network (CNN) model trained using TensorFlow/Keras.
2. A Flask API that serves the trained model to classify uploaded bike images.

---

## 🚀 Features

- CNN-based image classification using TensorFlow/Keras
- Flask API endpoint for prediction (`/predict`)
- Real-time confidence-based decision making (minimum 60% confidence)
- Data augmentation using `ImageDataGenerator`
- Visualization of model training history

---

## 📁 Dataset Structure

The dataset folder `project/` should have the following structure:

```
project/
├── BikeType1/
│   ├── image1.jpg
│   ├── image2.jpg
├── BikeType2/
│   ├── image1.jpg
│   └── ...
└── ...
```

Each sub-folder represents a sports bike class.

---

## 🧠 Model Architecture

- Input Layer: 128x128x3 RGB image
- Conv2D -> MaxPooling -> Conv2D -> MaxPooling
- Flatten -> Dense -> Dropout -> Output

```python
model = Sequential([
    Input(shape=(128, 128, 3)),
    Conv2D(32, kernel_size=3, activation='relu'),
    MaxPooling2D(pool_size=2),
    Conv2D(64, kernel_size=3, activation='relu'),
    MaxPooling2D(pool_size=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])
```

---

## 🔧 Training

To train the model:

```bash
python train_model.py
```

It uses `ImageDataGenerator` for augmentation and `EarlyStopping` to avoid overfitting.

Model is saved as: `cnn_project2.h5`

---

## 🌐 Flask API

### Install Requirements

```bash
pip install flask tensorflow opencv-python numpy
```

### Run the API

```bash
python app.py
```

### Predict Endpoint

- **POST** `/predict`
- **Form-Data**: `image` (Upload .jpg/.png)
- **Returns**: JSON with class label and confidence

```json
{
  "class": "Ducati",
  "confidence": 0.93
}
```

If confidence is below 60%, it returns:

```json
{
  "class": "NOT FROM SPORTS BIKES",
  "confidence": 0.42
}
```

---

## 📊 Training Performance

Training history is visualized with plots of accuracy and loss.

---

## 📦 Files

- `train_model.py` - Training and saving the CNN model
- `app.py` - Flask API for inference
- `cnn_project2.h5` - Saved trained model
- `project/` - Dataset folder with sub-class folders

---

## 🔒 License

This project is for educational purposes.
