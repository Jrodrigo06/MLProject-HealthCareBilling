# **HealthCare Charges Prediction System**

This is a **HealthCare Charges Prediction System** developed using **Flask**, **Python**, and **Machine Learning** to predict healthcare charges based on various personal and lifestyle factors. The project uses multiple machine learning models trained on healthcare billing data and serves predictions through a web application.

## **Project Overview**

This project aims to predict the amount of healthcare charges for individuals based on features such as:

- **Age**
- **Sex**
- **BMI (Body Mass Index)**
- **Smoker status**
- **Region**

The web application is built using **Flask**, and the system includes a range of machine learning models to make predictions. Users can interact with the web app, input their data, and receive predicted healthcare charges. The models used in the project are:

- **Random Forest**
- **Decision Tree**
- **Gradient Boosting**
- **Linear Regression**
- **CatBoost Regressor**
- **AdaBoost Regressor**

### **Feature Engineering**
- **BMI * Smoker Binary Feature**: One of the key feature engineering steps in the project was to create an interaction term by multiplying BMI with the smoker status (`smoker_binary`). This is because smoking status and BMI seem to have a combined effect on healthcare charges while just bmi didn't.
- **Removed Irrelevant Features**: Features that showed little to no correlation with healthcare charges were removed to simplify the model and improve performance. These features were identified through exploratory data analysis (EDA) and correlation checks.

### **Hyperparameter Tuning**
To improve the performance of the machine learning models, **GridSearchCV** was used for hyperparameter tuning. GridSearchCV performs an exhaustive search over a specified parameter grid to find the best combination of hyperparameters for each model. This process ensures that the models are optimized for better predictions.

## **Key Features**

- **Web Interface**: The system provides an intuitive web interface for users to input their data (age, sex, BMI, smoking status, and region).
- **Multiple ML Models**: The system uses a variety of machine learning models for making predictions, which gives flexibility in performance and comparison.
- **Customizable Data Preprocessing**: The input data undergoes preprocessing to ensure it is ready for model inference. This includes scaling numerical data and creating new interaction features.
- **Model Deployment**: The trained models are integrated into the Flask web application, providing real-time predictions.
- **Cross-Platform Compatibility**: Works on both Windows and Mac machines, with platform-specific configurations managed.

## **Technologies Used**

- **Flask**: Web framework for building the application and handling user requests.
- **Python**: The primary language used for data processing, machine learning, and web development.
- **Scikit-learn**: For preprocessing, model evaluation, and hyperparameter tuning with GridSearchCV.
- **CatBoost**: Used as one of the models for making predictions.
- **AdaBoost, GradientBoosting, Random Forest, Decision Tree, Linear Regression**: Other machine learning models included in the system.
- **HTML**: Used to create the front-end interface.

## **Project Setup and Installation**

### **1. Clone the Repository**

Clone the repository to your local machine using the following command:

```bash
git clone https://github.com/your-username/HealthCareChargesPrediction.git

## Mac
python3 -m venv venv
source venv/bin/activate

## Windows
python -m venv venv
.\venv\Scripts\activate

##Bash/cmd
pip install -r requirements.txt

##Run the app
python app.py


