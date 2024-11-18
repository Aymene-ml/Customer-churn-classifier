# Customer Churn Classifier

This project leverages machine learning to  predict customer churn based on diffent feature. The model is deployed using Flask and render, providing a simple a web user interface.

## Features

- **Customer Churn Classification**: The web app takes the following data  and predicts whether the customer is likely to churn:
    - **customerID**: Unique identifier for the customer.  
    - **gender**: Gender of the customer (Male/Female).  
    - **SeniorCitizen**: Indicates whether the customer is a senior citizen (1 for Yes, 0 for No).  
    - **Partner**: Indicates if the customer has a partner (Yes/No).  
    - **Dependents**: Indicates if the customer has dependents (Yes/No).  
    - **tenure**: Number of months the customer has stayed with the company.  
    - **PhoneService**: Indicates if the customer has phone service (Yes/No).  
    - **MultipleLines**: Indicates if the customer has multiple lines (Yes/No/No phone service).  
    - **InternetService**: Type of internet service provider (DSL/Fiber optic/No).  
    - **OnlineSecurity**: Indicates if the customer has online security (Yes/No/No internet service).  
    - **OnlineBackup**: Indicates if the customer has online backup (Yes/No/No internet service).  
    - **DeviceProtection**: Indicates if the customer has device protection (Yes/No/No internet service).  
    - **TechSupport**: Indicates if the customer has tech support (Yes/No/No internet service).  
    - **StreamingTV**: Indicates if the customer has streaming TV (Yes/No/No internet service).  
    - **StreamingMovies**: Indicates if the customer has streaming movies (Yes/No/No internet service).  
    - **Contract**: Type of contract term (Month-to-month/One year/Two year).  
    - **PaperlessBilling**: Indicates if the customer has paperless billing (Yes/No).  
    - **PaymentMethod**: Customer’s payment method (Electronic check/Mailed check/Bank transfer (automatic)/Credit card (automatic)).  
    - **MonthlyCharges**: Monthly charges incurred by the customer.  
    - **TotalCharges**: Total charges incurred by the customer.  
    - **Churn**: Target variable indicating whether the customer churned (Yes/No).  
- **Web Interface**: A form-based user interface for submitting data and receiving predictions.

## Technologies Used

- **Flask**: A lightweight Python web framework used to deploy the machine learning model and serve the web interface and API.
- **Scikit-Learn**: Machine learning library used to train the model, specifically using a Gradient Boosting algorithm for predicting power consumption.
- **dill**: A Python module used to serialize the trained machine learning model for deployment.
- **HTML/boostrap**: For creating the frontend user interface.
- **render**: To deploy the model to the cloud.

## Setup Instructions

1. **Clone the repository**:

    ```bash
    git clone <repository_url>
    cd <repository_folder>
    ```

2. **Set up a virtual environment**:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

4. **Run the Flask application**:

    ```bash
    flask run
    ```

    The app will be available at `http://127.0.0.1:5000`.

## How to Use

### Web Interface

- Open the web app in your browser.
- Navigate to the provided URL: [https://customer-churn-classifier.onrender.com](https://customer-churn-classifier.onrender.com).

## Model Details

The machine learning model is trained using diffent customers' information like demographics,services and payments.

The model was trained using a Support Vector classifier (SVC), which was serialized using dill and is loaded for use in the app.

  
