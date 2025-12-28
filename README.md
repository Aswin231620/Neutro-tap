# 🧠 NeuroTap – Parkinson Detection Using Voice

NeuroTap is a Machine Learning–based web application that detects the risk of Parkinson’s disease using voice features.

## 🚀 Features
- CSV-based voice feature prediction
- Direct `.wav` voice upload
- Risk level & confidence score
- Disease severity estimation
- PDF report generation
- Streamlit-based UI

## 🛠 Tech Stack
- Python
- Scikit-learn
- Librosa
- Streamlit
- NumPy, Pandas

## 📦 Model Files

The trained model (`parkinson_model.pkl`) and scaler (`scaler.pkl`) are not included in this repository.

To run the app:
- Train the model using the provided notebook (or your own dataset)
- Place the generated `.pkl` files in the project root

This follows standard ML deployment practices.

## ▶️ How to Run
```bash
pip install -r requirements.txt
streamlit run app.py

