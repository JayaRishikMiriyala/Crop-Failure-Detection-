import tensorflow as tf
import numpy as np
import sys
import cv2

def run_prediction(image_path):
    # 1. Load Model
    try:
        model = tf.keras.models.load_model('../models/crop_model.h5')
    except:
        print("Error: Model file not found. Run train.py first.")
        return

    # 2. Preprocess Image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img_array = np.expand_dims(img, axis=0) / 255.0

    # 3. Predict
    prediction = model.predict(img_array)[0][0]
    
    if prediction > 0.5:
        print(f"RESULT: [FAILURE DETECTED] (Confidence: {prediction:.2f})")
        print("Action Recommended: Inspect for pests or irrigation issues.")
    else:
        print(f"RESULT: [HEALTHY CROP] (Confidence: {1-prediction:.2f})")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py path/to/your/image.jpg")
    else:
        run_prediction(sys.argv[1])
