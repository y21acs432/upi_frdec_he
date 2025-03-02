import pandas as pd
from sklearn.model_selection import train_test_split
from model import PrivateXGBoostModel
from config import DATASET_PATH

df = pd.read_csv(DATASET_PATH)
X = df.drop('is_fraud', axis=1)
y = df['is_fraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

private_model = PrivateXGBoostModel()
private_model.train_model(X_train, y_train)
private_model.encrypt_model_weights()
private_model.save_model()
