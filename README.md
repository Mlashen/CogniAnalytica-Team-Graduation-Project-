
# 🫀 Heart Attack Risk Prediction System — CogniAnalytica

## 📌 Project Overview

**CogniAnalytica** is a graduation project focused on building a smart and interactive system for predicting the risk of heart attacks using machine learning and data analysis. The system utilizes various health and demographic indicators to provide accurate risk predictions, supporting both individuals and healthcare professionals in making better health decisions.

### 🎯 Project Goals

- **Academic Application**: Apply knowledge in statistics, data science, and programming to address a real-world health issue.
- **Model Development**: Build a reliable machine learning model for heart attack prediction using personal health data.
- **Technical Advancement**: Leverage tools like Python, Power BI, and Streamlit for data processing, modeling, and visualization.
- **Real-world Value**: Deliver a tool that offers actionable insights to assist in early diagnosis and preventive care.


###The data used for this project was sourced from Kaggle and is titled Personal Key Indicators of Heart Disease. This dataset contains valuable health information, such as age, sex, cholesterol levels, blood pressure, and other personal health indicators. Size: Approximately 20 MB (this may vary based on the version). The dataset can be accessed directly via this link:https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease
Kaggle Dataset 🌐
---

## 💻 Setup Instructions (Windows & macOS)

### 1. 🔗 Download the Project

Download the complete project from the following Google Drive link:

📂 [Google Drive - CogniAnalytica Project](https://drive.google.com/drive/folders/1bPIwApX2oGkWEKph1HLxEHdnLK0qfTla)

Once downloaded, extract the zip file and save it in your desired location.

---

### 2. 🧰 Prerequisites

| Tool / Library         | Required Version     |
|------------------------|----------------------|
| Python                 | 3.8 or later         |
| pip (Python package manager) | Latest version |
| Power BI Desktop       | (Optional) Latest version for dashboard viewing |

---

### 3. 📦 Create Virtual Environment and Install Dependencies

**Create a virtual environment:**

```bash
python -m venv venv
````

**Activate the virtual environment:**

* **Windows:**

  ```bash
  venv\Scripts\activate
  ```
* **macOS/Linux:**

  ```bash
  source venv/bin/activate
  ```

**Install required Python packages:**

```bash
pip install -r requirements.txt
```

> If `requirements.txt` is missing, you can manually install dependencies:

```bash
pip install streamlit pandas numpy seaborn matplotlib scikit-learn joblib
```

---

### 4. 🚀 Run the Application

Navigate to the user interface folder:

```bash
cd "Machine Code & User interface"
streamlit run app.py
```

The app will automatically open in your default web browser at `http://localhost:8501/`.

---

### 5. 🧪 How to Use the App

Once the app launches:

* Input your health data in the sidebar:

  * Age
  * Gender
  * BMI
  * Smoking and alcohol use
  * Sleep hours
  * General health status
  * Number of permanent teeth removed
* Click the **Predict** button to see your heart attack risk level.

---

### 6. 📊 Visual Analytics (Power BI Dashboards)

If you want to explore advanced visual analytics:

* Open the file `graduation project.pbix` located inside the `Power BI (Dashboards)/` folder using **Power BI Desktop**.
* Interact with charts showing:

  * Heart attack distribution by age, gender, health habits
  * Key factors influencing risk
  * Geographical and demographic insights

---

## ⚙️ Project Structure

```
Graduation Project/
│
├── GRADUATION PROJECT (Data Analysis).ipynb
├── heart_2022_with_nans.csv
├── heart_2022_with_nans_cleaned.xlsx
├── heart_2022_with_nans_edit (cleaned).ipynb
│
├── Machine Code & User interface/
│   ├── app.py
│   ├── best_heart_model.pkl
│   ├── heart_2022_no_nans.csv
│   ├── interFace.ipynb
│   ├── Machine Model.ipynb
│   ├── label_encoders_heart_attack.joblib
│   ├── model_features.pkl
│   ├── target_encoder.pkl
│
└── Power BI (Dashboards)/
    └── graduation project.pbix
```

---

## 🧩 Common Issues & Troubleshooting

| Issue                                      | Solution                                                                                                                                              |
| ------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ModuleNotFoundError` or missing libraries | Run `pip install -r requirements.txt` again or manually install missing packages                                                                      |
| App doesn’t open automatically in browser  | Manually go to `http://localhost:8501` in your browser                                                                                                |
| Model files not found error                | Ensure these files exist in the same folder: `best_heart_model.pkl`, `label_encoders_heart_attack.joblib`, `model_features.pkl`, `target_encoder.pkl` |
| Power BI visuals not displaying            | Ensure Power BI Desktop is properly installed and you're opening the `.pbix` file directly                                                            |

---
📚 **Project Documentation & User Guide:**  
Find detailed documentation, setup instructions, and user guide on GitBook:  
[CogniAnalytica Team Graduation Project Documentation](https://cognianalytica.gitbook.io/cognianalytica-team-graduation-project/)




