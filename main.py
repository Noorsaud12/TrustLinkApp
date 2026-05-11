import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. تحميل البيانات
path_real = r"C:\Users\lenovo\OneDrive\Desktop\project\real_users.csv"
path_fake = r"C:\Users\lenovo\OneDrive\Desktop\project\fake_users.csv"

real_df = pd.read_csv(path_real)
fake_df = pd.read_csv(path_fake)

# 2. إضافة التصنيف
real_df["label"] = 1
fake_df["label"] = 0

# 3. دمج البيانات
df = pd.concat([real_df, fake_df], ignore_index=True)

# 4. حذف الأعمدة غير المهمة
df = df.drop(columns=["id"], errors="ignore")

# 5. تحديد X و y
X = df.drop("label", axis=1)
y = df["label"]

# 6. تحويل كل القيم النصية إلى أرقام
for column in X.columns:
    X[column] = X[column].astype(str)
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])

# 7. تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 8. إنشاء وتدريب النموذج
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 9. التنبؤ
y_pred = model.predict(X_test)

# 10. النتائج
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 🔽 هنا تضيف الحفظ
import pickle

pickle.dump(model, open("model.pkl", "wb"))
print("Model saved successfully 👍")