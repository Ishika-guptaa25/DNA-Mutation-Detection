import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense

# =========================
# LOAD DATA
# =========================
train_df = pd.read_csv("DATA/train.csv")
val_df = pd.read_csv("DATA/validation.csv")

# Main DNA column
train_seq = train_df["NucleotideSequence"].astype(str).str.upper()
val_seq = val_df["NucleotideSequence"].astype(str).str.upper()

# =========================
# k-mer ENCODING (ADVANCED)
# =========================
def kmer(seq, k=3):
    return [seq[i:i+k] for i in range(len(seq)-k+1)]

train_kmers = train_seq.apply(kmer)
val_kmers = val_seq.apply(kmer)

# Encode kmers
all_kmers = sorted(set(k for seq in train_kmers for k in seq))
encoder = LabelEncoder()
encoder.fit(all_kmers)

X_train = train_kmers.apply(lambda x: encoder.transform(x))
X_val = val_kmers.apply(lambda x: encoder.transform(x))

# Padding
X_train = pad_sequences(X_train, maxlen=300)
X_val = pad_sequences(X_val, maxlen=300)

# Dummy labels (sequence learning task)
y_train = np.ones(len(X_train))
y_val = np.ones(len(X_val))

# =========================
# ADVANCED MODEL
# =========================
model = Sequential([
    Embedding(input_dim=len(all_kmers) + 1, output_dim=128),
    Conv1D(128, kernel_size=5, activation="relu"),
    MaxPooling1D(),
    Bidirectional(LSTM(64)),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# =========================
# TRAIN MODEL
# =========================
model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=8,
    batch_size=32
)

# Save model
model.save("dna_advanced_model.h5")

print("✅ Model trained & saved as dna_advanced_model.h5")
