import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import pandas as pd

st.set_page_config(page_title="DNA Mutation Detection", layout="centered")

st.title("🧬 Advanced DNA Sequence Analysis System")

# Load model
model = load_model("dna_advanced_model.h5")

# Load training data again to rebuild encoder
train_df = pd.read_csv("DATA/train.csv")
train_seq = train_df["NucleotideSequence"].astype(str).str.upper()

def kmer(seq, k=3):
    return [seq[i:i+k] for i in range(len(seq)-k+1)]

train_kmers = train_seq.apply(kmer)
all_kmers = sorted(set(k for seq in train_kmers for k in seq))

encoder = LabelEncoder()
encoder.fit(all_kmers)

# =========================
# USER INPUT
# =========================
dna = st.text_area(
    "Enter DNA Sequence (A, T, G, C only)",
    placeholder="ATGCTAGCTAGCTAACG"
)

if st.button("Analyze Sequence"):
    dna = dna.upper().strip()

    if len(dna) < 20:
        st.warning("⚠️ DNA sequence too short")
    else:
        kmers = kmer(dna)
        encoded = encoder.transform([k for k in kmers if k in encoder.classes_])
        padded = pad_sequences([encoded], maxlen=300)

        prediction = model.predict(padded)[0][0]

        st.success("✅ Sequence analyzed successfully")
        st.info(f"🔬 Pattern Score: {prediction:.4f}")

        st.markdown(
            """
            **Model Used:**  
            - k-mer feature extraction  
            - CNN + BiLSTM hybrid deep learning  
            - Trained on real DNA datasets  
            """
        )
