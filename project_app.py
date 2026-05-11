import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# =========================
# تحميل الموديل
# =========================
model = pickle.load(open("model.pkl", "rb"))

# =========================
# تحميل البيانات
# =========================
# تحميل البيانات
path_real = "real_users.csv"
path_fake = "fake_users.csv"

real_df = pd.read_csv(path_real)
fake_df = pd.read_csv(path_fake)
real_df = pd.read_csv(path_real)
fake_df = pd.read_csv(path_fake)

real_df["label"] = 1
fake_df["label"] = 0

df = pd.concat([real_df, fake_df], ignore_index=True)
df = df.drop(columns=["id"], errors="ignore")

# =========================
# UI
# =========================
st.title("🔍 Fake vs Real User Detection System")

username = st.text_input("Enter Username")

if st.button("Search"):

    user_data = df[df["name"].astype(str).str.lower() == username.strip().lower()]

    if user_data.empty:
        st.error("❌ User not found in dataset")

    else:
        st.success("✅ User found")

        # =========================
        # اختيار الأعمدة المطلوبة
        # =========================
        st.subheader("📊 User Stats")

        # نجيب الأعمدة الرقمية فقط
        numeric_df = user_data.select_dtypes(include=['int64', 'float64'])

        if numeric_df.shape[1] == 0:
            st.warning("No numeric data available for chart")
        else:

            data = numeric_df.iloc[0]

            fig, ax = plt.subplots()

            ax.bar(data.index[:5], data.values[:5])

            ax.set_title("User Profile Stats")

            plt.xticks(rotation=45)

            st.pyplot(fig)

        # =========================
        # التنبؤ
        # =========================
        X = user_data.drop("label", axis=1)

        for col in X.columns:
            X[col] = X[col].astype(str)
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])

        prediction = model.predict(X)

        if prediction[0] == 1:
            st.success("✅ REAL USER")
        else:
            st.error("❌ FAKE USER")