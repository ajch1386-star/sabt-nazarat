import streamlit as st
import requests
import numpy as np
import pandas as pd
import joblib
import time

st.set_page_config(page_title="ویرایشگر نظر خریدار", layout="centered")

# ---------------------
# تنظیمات HuggingFace API
# ---------------------
HF_MODEL = "HooshvareLab/bert-fa-base-uncased-sentiment-snappfood"
API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

def get_hf_headers():
    key = st.secrets.get("HF_API_KEY")
    if not key:
        return None
    return {"Authorization": f"Bearer {key}"}

# ---------------------
# فراخوانی API تحلیل احساس
# ---------------------
def get_sentiment_from_api(text):
    headers = get_hf_headers()
    if headers is None:
        return None, "missing_api_key"

    try:
        response = requests.post(API_URL, headers=headers, json={"inputs": text}, timeout=20)
        data = response.json()

        # مدل در حال بارگذاری است
        if isinstance(data, dict) and "error" in data:
            if "loading" in data["error"].lower():
                return None, "loading"
            return None, "api_error"

        # خروجی صحیح
        if isinstance(data, list) and len(data) and isinstance(data[0], list):
            label = data[0][0]["label"]
            score = data[0][0].get("score", 0.0)
            return (label, score), None

        return None, "unexpected_response"

    except requests.exceptions.RequestException:
        return None, "network_error"
    except Exception as e:
        return None, f"unknown_error: {e}"


# ---------------------
# بارگذاری مدل joblib
# ---------------------
@st.cache_resource(show_spinner=False)
def load_model():
    try:
        model = joblib.load("mymodel.joblib")
        return model, None
    except FileNotFoundError:
        return None, "model_not_found"
    except Exception as e:
        return None, f"model_load_error: {e}"

model, model_err = load_model()

# ---------------------
# رابط کاربری
# ---------------------
st.title("🛍️ ویرایشگر نظر خریدار")
st.markdown("نظر را وارد کنید تا سیستم با تحلیل احساس و مدل تصمیم بگیرد که ثبت شود یا خیر.")

sent = st.text_area("نظر خود را وارد کنید:", height=130)
col1, col2, col3 = st.columns(3)

with col1:
    ops = st.slider("نقاط قوت", 0, 3, 1)
with col2:
    neg = st.slider("نقاط ضعف", 0, 3, 1)
with col3:
    score = st.slider("امتیاز کالا", 1, 5, 3)


# ---------------------
# بررسی مدل
# ---------------------
if model_err:
    st.error("❌ مدل mymodel.joblib بارگذاری نشد. فایل را در ریشه پروژه قرار بده.")
    st.stop()


# ---------------------
# اجرای تحلیل
# ---------------------
if st.button("🔍 تحلیل و ثبت نظر"):
    if sent.strip() == "":
        st.warning("متن نظر را وارد کنید.")
        st.stop()

    st.info("⏳ در حال تحلیل احساس...")

    # مدل HF معمولاً 5–10 ثانیه warmup لازم دارد
    for _ in range(7):   # حدوداً 20 ثانیه
        result, err = get_sentiment_from_api(sent)

        if err == "loading":
            st.write("🟡 مدل در حال آماده‌سازی است... لطفاً چند لحظه صبر کنید.")
            time.sleep(3)
            continue

        break

    # خطاهای مختلف
    if err == "missing_api_key":
        st.error("❌ کلید HuggingFace در Secrets تنظیم نشده.")
        st.stop()

    if err == "loading":
        st.error("❌ مدل هنوز آماده نیست. چند ثانیه دیگر امتحان کنید.")
        st.stop()

    if err is not None:
        st.error(f"❌ خطا در تحلیل احساس: {err}")
        st.stop()

    # نتیجه موفق
    label, conf = result
    des = 1 if label == "HAPPY" else 0

    st.success(f"احساس متن: **{label}** (اعتماد: {conf:.1%})")

    # ---------------------
    # پیش‌بینی با مدل sklearn
    # ---------------------
    raw = np.array([[des, ops, neg]])
    x = pd.DataFrame(raw, columns=["des", "ops", "neg"])

    try:
        pre = int(model.predict(x)[0])
    except Exception as e:
        st.error(f"❌ خطا در اجرای مدل: {e}")
        st.stop()

    # ---------------------
    # تصمیم نهایی
    # ---------------------
    if pre == 1 and score >= 3:
        st.success("✅ نظر شما ثبت شد")
    elif pre == 1 and score < 3:
        st.error("❌ امتیاز کافی نیست. نظر ثبت نشد")
    elif pre == 0 and score < 3:
        st.success("✅ نظر شما ثبت شد")
    else:
        st.error("❌ امتیاز نامناسب. نظر ثبت نشد")


st.markdown("---")
st.markdown("📌 **نکته:** مقدار `HF_API_KEY` باید در Settings → Secrets تنظیم شود.")
