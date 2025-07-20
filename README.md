# 🫀 Heart Disease Prediction Web App using Streamlit
## Link to access the deployed web app of this project 
[https://heart-disease-prediction-application.streamlit.app/](https://jappanjot-heart-disease-prediction-app.streamlit.app/)

This project is a **machine learning-powered web application** built using **Streamlit** to predict the likelihood of heart disease based on medical attributes. It enables users to input health data and receive an instant prediction about their heart health status.

---

## 📌 Features

- 🌐 Simple and interactive **web interface** using Streamlit  
- 🤖 Trained **Logistic Regression** model for binary classification  
- 📊 Real-time prediction of heart disease likelihood  
- ✅ Clean UI for inputting 13 health-related parameters  
- 💡 Includes model training and evaluation scripts  

---

## 📁 Project Structure

```
Heart-Disease-Streamlit-App/
├── app.py     # Main Streamlit app file
├── heart_disease_data.csv          # Dataset file
├── requirements.txt                # Project dependencies
└── README.md                       # Project documentation
```

---

## ⚙️ Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/jappanjotkaur/Heart-Disease-Streamlit-App.git
   cd Heart-Disease-Streamlit-App
   ```

2. **(Optional) Create and Activate a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the App**
   ```bash
   streamlit run app.py
   ```

---

## 🧪 Input Features

The app requires the following 13 input features:

- Age  
- Sex  
- Chest Pain Type  
- Resting Blood Pressure  
- Serum Cholesterol  
- Fasting Blood Sugar  
- Resting ECG  
- Maximum Heart Rate  
- Exercise-Induced Angina  
- ST Depression  
- Slope of Peak ST  
- Number of Major Vessels  
- Thalassemia  

---

## 📊 Output

- **Result**: `"Patient has heart disease"` or `"Patient does not have heart disease"`  
- **Probability Score**: Based on logistic regression model output  

---

