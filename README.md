# ❤️ Heart Disease Predictor

Streamlit Link - https://heart-disease-predictor-scratch.streamlit.app/

A machine-learning web app built using **PyTorch**, **Logistic Regression**, **SVM**, and **Streamlit** to predict heart disease risk.

This project includes saved model parameters, scaler, Streamlit UI, and features like **risk gauge**, **PDF health report**, and **doctor recommendations**.

✔️ Predict heart disease risk   
✔️ Doctor recommendation insights  
✔️ Downloadable PDF health report  
✔️ Clean card-based UI for inputs  

---

## 🏗️ File Structure
```
heart-disease-predictor/
├── app.py # Streamlit Application
├── models.pth # Saved PyTorch models (Logistic Regression + SVM)
├── scaler.pkl # StandardScaler object
├── requirements.txt # Python dependencies
└── README.md # Project documentation
```

---

## 🔧 Installation & Setup (All-in-One)

```bash
# 1. Clone the repository
git clone https://github.com/amanverma-98/AlgofromScratch_with_PyTorch.git
cd heart-disease-predictor

# 2. Create & activate virtual environment
python -m venv env
# Linux/Mac
source env/bin/activate
# Windows
env\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Streamlit app
streamlit run app.py
```


🧩 **How It Works**
```

🔹 Preprocessing
Input values are scaled using scaler.pkl
Categorical variables are one-hot encoded internally

🔹 Prediction
Logistic Regression: High , Moderate , Low Risk
SVM: outputs -1 or 1 for risk prediction

🔹 PDF Report
Generated using FPDF
Contains all inputs, predictions, and doctor advice
Includes disclaimer
```

📦 **Required Files**

```
models.pth	Saved model weights for Logistic Regression & SVM
scaler.pkl	StandardScaler for input features
app.py	Streamlit UI and prediction logic
requirements.txt	Python dependencies
```


📝 **Future Improvements**
```
Add more real-world features for more accurate predictions
Integrate transformer models for enhanced risk estimation
Improve UI styling with modern animations and interactive charts
Include user authentication & history tracking
Add real-time model retraining with new incoming data
```

⚠️ **Disclaimer**
```
This tool is for informational purposes only and does not constitute a medical diagnosis.
Always consult a qualified healthcare professional for personalized advice.
```

🤝 **Contributing**

Contributions are welcome—open issues or PRs for suggestions or improvements.


📜 **License**
```
Released under the MIT License.

This is **truly everything in a single block**:  
- File structure  
- Installation + setup commands  
- How it works  
- Required files  
- `requirements.txt`  
- Future improvements  
``` 

