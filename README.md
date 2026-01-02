## DNA Mutation Detection System
### Live Application

https://dna-mutation-detection-jtytqvvj4bwqffjpm9aaoj.streamlit.app/

### Abstract

This project implements a deep learning–based system for DNA sequence analysis and mutation pattern detection.
It combines bioinformatics feature extraction techniques with neural network modeling to analyze nucleotide sequences computationally.
The trained model is deployed as a web application using Streamlit, enabling real-time sequence analysis through a browser-based interface.

### Objectives

To analyze DNA nucleotide sequences using computational methods

To apply k-mer based feature extraction for genomic data

To train a deep learning model capable of learning sequence patterns

To deploy the trained model as an interactive web application

To demonstrate practical integration of bioinformatics and machine learning

### System Description

The system accepts DNA sequences composed of nucleotides A, T, G, and C.
Each sequence is transformed into overlapping k-mers to preserve biological context.
These k-mers are encoded numerically and processed by a hybrid CNN–LSTM model that captures both local sequence motifs and long-range dependencies.

The final output represents a learned pattern score derived from the trained model.

### Methodology

DNA sequence input is provided by the user

Sequence is segmented into overlapping k-mers

k-mers are encoded into numerical form

Encoded sequences are padded to a fixed length

CNN layers extract local genomic features

LSTM layer captures sequential dependencies

Model produces an analysis score

### Dataset Description

Format: CSV

Key Column: NucleotideSequence

Data Splits:

Training dataset

Validation dataset

Test dataset

Contains real DNA nucleotide sequences used for supervised learning

### Model Architecture

Embedding Layer

1D Convolution Layer

Max Pooling Layer

LSTM Layer

Fully Connected Dense Layer

This architecture enables efficient learning of genomic sequence patterns.

### Project Structure
<img width="600" height="350" alt="image" src="https://github.com/user-attachments/assets/7e3689dc-a776-4301-bfbf-572723b7730c" />

### Demo Screenshot 
<img width="800" height="609" alt="image" src="https://github.com/user-attachments/assets/7a91e469-7dee-4e78-b812-ce30c0c92105" />

### Example Input
ATGCTAGCTAGCTAACGATGCTAG

### Technology Stack

Python

TensorFlow / Keras

NumPy

Pandas

Scikit-learn

Streamlit

### Deployment

The system is deployed using Streamlit Cloud, providing an accessible and lightweight platform for real-time inference.

Live URL:
https://dna-mutation-detection-jtytqvvj4bwqffjpm9aaoj.streamlit.app/

### Use Cases

Academic bioinformatics projects

DNA sequence pattern analysis

Deep learning portfolio demonstration

Internship and research evaluations

Forking and Contribution Guidelines

This repository is open for educational and academic use.

### How to Fork

#### Open the repository on GitHub

Click the Fork button in the top-right corner

A copy of the repository will be created under your GitHub account

#### Cloning a Fork
git clone https://github.com/ishika-guptaa25/DNA-Mutation-Detection.git

cd DNA-Mutation-Detection

#### Making Contributions

Create a new branch

git checkout -b feature-name


#### Commit changes

git commit -m "Describe changes clearly"


#### Push to your fork

git push origin feature-name


#### Open a Pull Request with a clear description

Academic Integrity Notice

If this repository is forked for academic submission:

Proper attribution to the original repository is required

The project should be used as a learning reference

Direct submission without understanding may violate institutional policies

### Limitations

The current system focuses on sequence pattern learning rather than explicit mutation type classification

FASTA file input is not yet supported

Model performance depends on dataset size and quality

### Future Enhancements

Explicit mutation classification (insertion, deletion, substitution)

FASTA and batch sequence support

Model evaluation metrics such as accuracy and confusion matrix

Cancer genomics datasets

Transformer-based genomic models
