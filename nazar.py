import streamlit as st
import requests
import numpy as np
import pandas as pd
import joblib

# ---------------------
# HuggingFace API Setup
# ---------------------
API_URL = "https://api-inference.huggingface.co/models/HooshvareLab/bert-fa-base-uncased-sentiment-snappfood"
headers = {"Authorization": f"Bearer {st.secrets['HF_API_KEY']}"}

def get_sentiment(text):
    res = requests.post(API_URL, headers=headers, json={"inputs": text}).json()
    try:
        return res[0][0]["label"]
    except:
        return "SAD"  # fallback if HuggingFace returns empty response

# ---------------------
# Load sklearn model
# ---------------------
model = joblib.load("mymodel.joblib")

# ---------------------
# Streamlit UI
# ---------------------
st.title('ویرایشگر نظر خریدار')

sent = st.text_input('نظر خریدار را وارد کنید')
ops = st.slider('تعداد نقاط قوت', 1, 3, 1)
neg = st.slider('تعداد نقاط ضعف', 1, 3, 1)
score = st.slider('امتیاز کالا (۱ تا ۵)', 1, 5, 3)

if st.button("ثبت نظر"):
    label = get_sentiment(sent)
    des = 1 if label == "HAPPY" else 0

    raw = np.array([[des, ops, neg]])
    x = pd.DataFrame(raw, columns=['des', 'ops', 'neg'])

    pre = int(model.predict(x)[0])

    if pre == 1 and score >= 3:
        st.success("نظر شما ثبت شد")
    elif pre == 1 and score < 3:
        st.error("امتیاز کافی نیست")
    elif pre == 0 and score < 3:
        st.success("نظر شما ثبت شد")
    else:
        st.error("امتیاز مناسب نیست؛ نظر ثبت نشد")
