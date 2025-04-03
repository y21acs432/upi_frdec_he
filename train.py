import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, classification_report,roc_auc_score,precision_recall_curve,auc,confusion_matrix,ConfusionMatrixDisplay
import logging
from imblearn.over_sampling import SMOTE,ADASYN
from imblearn.combine import SMOTEENN
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from model import PrivateXGBoostModel
from config import DATASET_PATH,LABLE_PATH

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load dataset
logger.info("Loading dataset...")
df = pd.read_csv(DATASET_PATH)


# Drop unwanted columns (if they exist)
df = df.drop(columns=["transaction_time", "upi_id"], errors="ignore")


# Encode categorical features
logger.info("Encoding categorical features...")
encoders = {}
for col in ["location", "device_type", "merchant_category"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le  # Store encoders for later use

with open(LABLE_PATH, "wb") as f:
    pickle.dump(encoders, f)
logger.info("Label encoders saved successfully.")

# Define features and target variable
X = df.drop(columns=["is_fraud"])
y = df["is_fraud"]

# Split into training and testing sets (Stratified split to maintain class balance)
logger.info("Splitting dataset into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

#smote = SMOTE(sampling_strategy=0.5, random_state=42)
#X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

#undersampler = RandomUnderSampler(sampling_strategy=0.2, random_state=42)
#X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train, y_train)

# Check new fraud distribution
#print("Fraud Count After SMOTE:\n", pd.Series(y_train_resampled).value_counts(normalize=True))

counter = Counter(y_train)
print('Before', counter)

sme = SMOTEENN(random_state=42)
X_train_smtom, y_train_smtom = sme.fit_resample(X_train, y_train)

# Counting the number of instances in each class after oversampling
counter = Counter(y_train_smtom)
print('After', counter)

# Train the private XGBoost model
logger.info("Training the model...")
private_model = PrivateXGBoostModel()
private_model.train_model(X_train_smtom, y_train_smtom)

# Make predictions on the test set
logger.info("Making predictions...")
y_pred = private_model.model.predict(X_test)


precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred)

# Find the best threshold where precision and recall are balanced
f1_scores = 2 * (precisions * recalls) / (precisions + recalls)


# Calculate accuracy
#accuracy = accuracy_score(y_test, y_pred)
#logger.info(f"\nModel Training Completed! Accuracy: {accuracy:.4f}")

auc_score = roc_auc_score(y_test, y_pred)
print(f"ROC-AUC Score: {auc_score:.4f}")


#y_pred_proba = private_model.model.predict_proba(X_test)[:, 1]  # Probability of fraud

precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
pr_auc = auc(recall, precision)



plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f'Precision-Recall curve (AUC = {pr_auc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.grid(True)
plt.show()


# Compute and display confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title('Confusion Matrix')
plt.show()


# Evaluate performance with Precision, Recall, and F1-score
#print("\nClassification Report (Train Data):\n", classification_report(y_train_balanced, private_model.model.predict(X_train_balanced)))
logger.info("\nClassification Report:\n" + classification_report(y_test, y_pred, zero_division=1))
y_train_pred = private_model.model.predict(X_train)
# Encrypt model weights and save the trained model
logger.info("Encrypting model weights and saving the trained model...")
private_model.encrypt_model_weights()
private_model.save_model()

logger.info("Training process completed successfully.")
