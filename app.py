# app.py
import streamlit as st
import requests
import numpy as np
import pandas as pd
import joblib

st.set_page_config(page_title="ویرایشگر نظر خریدار", layout="centered")

# ---------------------
# تنظیمات HuggingFace Inference API
# ---------------------
HF_MODEL = "HooshvareLab/bert-fa-base-uncased-sentiment-snappfood"
API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

# خواندن کلید از secrets (باید در Streamlit Cloud اضافه شود)
def get_hf_headers():
    key = st.secrets.get("HF_API_KEY")
    if not key:
        return None
    return {"Authorization": f"Bearer {key}"}

def get_sentiment_from_api(text):
    """
    فراخوانی HuggingFace Inference API برای تحلیل احساس
    بازگشت: برچسب رشته‌ای مثل "HAPPY" یا "SAD" یا None در صورت خطا
    """
    headers = get_hf_headers()
    if headers is None:
        return None, "missing_api_key"

    try:
        response = requests.post(API_URL, headers=headers, json={"inputs": text}, timeout=20)
        response.raise_for_status()
        data = response.json()
        # ساختار معمول خروجی: [[{"label":"HAPPY","score":0.98}, ...]]
        if isinstance(data, dict) and data.get("error"):
            return None, "api_error"
        # بعضی اوقات API مستقیماً لیست بازمی‌گرداند
        if isinstance(data, list) and len(data) and isinstance(data[0], list):
            label = data[0][0].get("label")
            score = data[0][0].get("score", 0.0)
            return (label, score), None
        # fallback
        return None, "unexpected_response"
    except requests.exceptions.RequestException:
        return None, "network_error"
    except Exception:
        return None, "unknown_error"

# ---------------------
# بارگذاری مدل scikit-learn (joblib)
# ---------------------
@st.cache_resource(show_spinner=False)
def load_sklearn_model(path="mymodel.joblib"):
    try:
        m = joblib.load(path)
        return m, None
    except FileNotFoundError:
        return None, "model_not_found"
    except Exception as e:
        return None, f"model_load_error: {e}"

model, model_err = load_sklearn_model()

# ---------------------
# رابط کاربری
# ---------------------
st.title("🛍️ ویرایشگر نظر خریدار")
st.markdown(
    "نظر خود را وارد کنید؛ سیستم احساس متن را تحلیل می‌کند و سپس با مدل تصمیم می‌گیرد که نظر ثبت شود یا خیر."
)

sent = st.text_area("نظر خود را وارد کنید", height=120, placeholder="مثال: این محصول کیفیت خوبی داشت...")
col1, col2, col3 = st.columns([1,1,1])
with col1:
    ops = st.slider("تعداد نقاط قوت", 0, 3, 1)
with col2:
    neg = st.slider("تعداد نقاط ضعف", 0, 3, 1)
with col3:
    score = st.slider("امتیاز کالا (۱ تا ۵)", 1, 5, 3)

st.write("")  # spacer

# نمایش هشدارها / وضعیت
if model_err is not None:
    st.error("مدل پیش‌بینی (mymodel.joblib) بارگذاری نشد. لطفاً فایل مدل را در ریشهٔ repo آپلود کن.")
    st.stop()

if st.button("🔍 تحلیل و ثبت نظر"):
    if not sent or sent.strip() == "":
        st.warning("لطفاً ابتدا متن نظر را وارد کنید.")
    else:
        with st.spinner("تحلیل احساس..."):
            result, err = get_sentiment_from_api(sent)
        if err == "missing_api_key":
            st.error("کلید HuggingFace API در Secrets تنظیم نشده. در Streamlit Cloud به Settings → Secrets برو و HF_API_KEY را قرار بده.")
            st.stop()
        if err is not None:
            st.error(f"خطا در تحلیل احساس ({err}). لطفاً بعداً امتحان کنید.")
            st.stop()
        label, conf = result
        des = 1 if label == "HAPPY" else 0

        st.info(f"تحلیل احساس: **{label}** (اطمینان: {conf:.2%})")

        # پیش‌بینی با مدل sklearn
        raw = np.array([[des, ops, neg]])
        x = pd.DataFrame(raw, columns=['des','ops','neg'])
        try:
            pre = int(model.predict(x)[0])
        except Exception as e:
            st.error(f"خطا در پیش‌بینی مدل: {e}")
            st.stop()

        # منطق ثبت
        if pre == 1 and score >= 3:
            st.success("✅ نظر شما ثبت شد")
        elif pre == 1 and score < 3:
            st.error("❌ امتیاز کافی نیست؛ نظر ثبت نشد")
        elif pre == 0 and score < 3:
            st.success("✅ نظر شما ثبت شد")
        else:
            st.error("❌ امتیاز مناسب نیست؛ نظر ثبت نشد")

# اطلاعات کمکی و لینک‌ها
st.markdown("---")
st.markdown("**نکات فنی:**")
st.write(
    "- برای کار با HuggingFace Inference API باید در Streamlit Cloud در Settings → Secrets مقدار `HF_API_KEY` را قرار دهید.\n"
    "- فایل `mymodel.joblib` باید در ریشهٔ repository قرار گیرد و با نسخهٔ numpy فضای اجرا (Streamlit Cloud) سازگار باشد.\n"
    "- اگر می‌خواهید داده‌ها را ذخیره کنید (مثلاً در دیتابیس یا Google Sheets)، می‌توانم آن را برایت اضافه کنم."
)
