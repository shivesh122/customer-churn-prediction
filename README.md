# 📊 Customer Churn Prediction

A machine learning project to predict which customers are likely to leave a company, with an interactive **Streamlit dashboard** for analysis, model training, and prediction.

---

## 🚀 Features
- 📂 Upload and preprocess customer datasets  
- 🔍 Perform **Exploratory Data Analysis (EDA)** with interactive charts  
- 🤖 Train and compare multiple ML models (Logistic Regression, Random Forest, XGBoost)  
- 📈 Visualize model performance with accuracy, precision, recall, and confusion matrix  
- 🧑‍💻 Predict churn for **single customers** or in **batch mode**  
- 💾 Save and load trained models for reuse  
- 🌐 Deployable directly on [Streamlit Cloud](https://streamlit.io/)  

---

## 📁 Project Structure
```
├── app.py                 # Main Streamlit app
├── data/                  # Sample dataset
├── models/                # Saved ML models
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

---

## 🛠️ Installation

1. Clone the repository  
   ```bash
   git clone https://github.com/your-username/customer-churn-prediction.git
   cd customer-churn-prediction
   ```

2. Create a virtual environment (optional but recommended)  
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies  
   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ Usage
Run the Streamlit app:
```bash
streamlit run app.py
```

This will start a local web server. Open the link shown in your terminal to use the dashboard.

---

## 📊 Example Dataset
We used the **Telco Customer Churn dataset** (publicly available on Kaggle).  
You can also upload your own dataset in `.csv` format.

---

## 🌟 Demo Screenshots
### Dashboard Preview
![Dashboard Screenshot](https://via.placeholder.com/900x400.png?text=Add+your+screenshot+here)

---

## 📌 Future Improvements
- Add deep learning models for churn prediction  
- Deploy as a REST API for integration with other apps  
- Add user authentication for multi-user dashboards  

---

## 🤝 Contributing
Contributions are welcome! Feel free to fork this repo, open issues, or submit pull requests.

---

## 📧 Contact
Created by **[Your Name](https://github.com/your-username)**  
For questions, reach out at *your.email@example.com*
