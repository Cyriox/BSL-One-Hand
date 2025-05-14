import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from tensorflow.keras.models import load_model

# Load dataset
df = pd.read_csv("dataset/dataset_numbers.csv")
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Split data (same as training)
_, X_test, _, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Load model and encoder classes
model = load_model("models/bsl_model.h5")
encoder.classes_ = np.load("models/label_encoder_classes.npy", allow_pickle=True)

# Predict and evaluate
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred_classes, target_names=[str(cls) for cls in encoder.classes_]))

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[str(cls) for cls in encoder.classes_],
            yticklabels=[str(cls) for cls in encoder.classes_])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()
