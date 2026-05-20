import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification

@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained("bert_model")
    model = BertForSequenceClassification.from_pretrained("bert_model")
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def predict(text):
    encoding = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    input_ids      = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits  = outputs.logits
        prob    = torch.softmax(logits, dim=1)[0]
        pred    = torch.argmax(prob).item()

    # Labels ulte hain — isliye swap kiya
    fake_prob = round(prob[1].item() * 100, 1)  # ← 0 se 1 kiya
    real_prob = round(prob[0].item() * 100, 1)  # ← 1 se 0 kiya

    return pred, fake_prob, real_prob
st.set_page_config(page_title="Fake News Detector", page_icon="🔍", layout="centered")
st.title("🔍 Fake News Detector")
st.markdown("### Powered by BERT — Context-Aware Detection!")
st.divider()

news = st.text_area("📰 News Text yahan paste karo:", height=200,
                    placeholder="Koi bhi news article yahan paste karo...")

col1, col2, col3 = st.columns([1,2,1])
with col2:
    check = st.button("🚀 Click Here ", use_container_width=True)

if check:
    if news.strip():
        word_count = len(news.split())
        if word_count < 20:
            st.warning(" Kam se kam 20 words ka article paste karo!")
        else:
            with st.spinner(" BERT Analyzing..."):
                pred, fake_prob, real_prob = predict(news)

            st.divider()
            if pred == 1:
                st.error("FAKE News")
                st.metric("Confidence", f"{fake_prob}%")
                st.progress(fake_prob / 100)
                if fake_prob > 90:
                    st.error("🚨 High Risk — Ye news bilkul fake hai!")
                elif fake_prob > 70:
                    st.warning("⚠️ Medium Risk — Verify karo!")
                else:
                    st.info("🤔 Low Risk — Thoda suspicious hai")
            else:
                st.success("✅ REAL News")
                st.metric("Confidence", f"{real_prob}%")
                st.progress(real_prob / 100)
                if real_prob > 90:
                    st.success("✅ Highly Credible News!")
                elif real_prob > 70:
                    st.info("📰 Likely Real — Cross verify recommended")
                else:
                    st.warning("🤔 Uncertain — Multiple sources check karo")
    else:
        st.warning("⚠️ Kuch toh paste karo pehle!")