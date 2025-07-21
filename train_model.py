import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
import joblib


df = pd.read_csv("adult.csv")


df.replace("?", pd.NA, inplace=True)
df.dropna(inplace=True)


label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    if col != "income":  
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le


target_le = LabelEncoder()
df["income"] = target_le.fit_transform(df["income"])  # <=50K → 0, >50K → 1


joblib.dump(target_le, "target_encoder.pkl")


X = df.drop("income", axis=1)
y = df["income"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)


joblib.dump(model, "knn_model.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")


print("✅ Model saved as knn_model.pkl")
print("✅ Encoders saved as label_encoders.pkl")
print("✅ Accuracy:", model.score(X_test, y_test))
