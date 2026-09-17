# Diabetes Risk Prediction Web Application

An educational machine-learning web application that uses a Random Forest classifier to estimate the probability of a positive diabetes outcome from eight input measurements.

The trained model is integrated into a Flask interface and prepared for deployment on Render.

## Live application

The deployed application will be available here:

**Render URL:** To be added after deployment

## Project objective

This project demonstrates the complete workflow of:

1. Exploring and preprocessing a dataset
2. Training and optimizing a classification model
3. Saving the preprocessing and modeling pipeline
4. Integrating the trained pipeline into a Flask application
5. Deploying the application as an online service

## Dataset

The project uses the Pima Indians Diabetes dataset. It contains 768 observations, eight predictor variables, and one binary target variable.

### Predictor variables

- `Pregnancies`
- `Glucose`
- `BloodPressure`
- `SkinThickness`
- `Insulin`
- `BMI`
- `DiabetesPedigreeFunction`
- `Age`

### Target

- `Outcome = 0`: Negative outcome
- `Outcome = 1`: Positive outcome

The dataset used in this project was obtained through the 4Geeks Academy dataset repository.

## Data preprocessing

Although the original dataset did not contain explicitly recorded null values, several physiologic measurements contained zeros that were treated as unrecorded values:

- Glucose
- Blood pressure
- Skin thickness
- Insulin
- BMI

A scikit-learn pipeline performs median imputation for these zero-coded missing measurements. The medians are learned only from the training data during model evaluation.

Feature scaling was not applied because Random Forest models do not depend on distances or coefficient magnitudes.

## Model development

The data were divided into stratified training and testing sets. A Random Forest classifier was evaluated using accuracy, precision, recall, F1-score, ROC-AUC, and a confusion matrix.

Hyperparameters were optimized with five-fold cross-validation using ROC-AUC as the scoring metric.

### Selected hyperparameters

```python
RandomForestClassifier(
    n_estimators=300,
    max_depth=6,
    min_samples_leaf=4,
    class_weight=None,
    random_state=42
)
```

### Cross-validation result

- Best cross-validated ROC-AUC: **0.841**

### Held-out test performance of the deployment pipeline

| Metric | Score |
|---|---:|
| Accuracy | 0.740 |
| Precision | 0.675 |
| Recall | 0.500 |
| F1-score | 0.574 |
| ROC-AUC | 0.807 |

After evaluation, the final deployment pipeline was fitted using the complete dataset and saved with Joblib.

## Web application

The Flask application:

1. Receives eight numeric values from an HTML form
2. Creates a one-row pandas DataFrame
3. Passes the data to the saved preprocessing-and-model pipeline
4. Generates a predicted class and estimated probability
5. Displays the result on a separate results page

## Repository structure

```text
flask-diabetes-predictor/
├── data/
│   └── diabetes.csv
├── models/
│   └── diabetes_pipeline.pkl
├── src/
│   ├── static/
│   │   └── style.css
│   ├── templates/
│   │   ├── index.html
│   │   └── result.html
│   ├── app.py
│   └── explore.ipynb
├── requirements.txt
└── README.md
```

## Running the application locally

Clone the repository and enter the project directory:

```bash
git clone <repository-url>
cd flask-diabetes-predictor
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Run with the Flask development server:

```bash
python src/app.py
```

Then visit:

```text
http://127.0.0.1:5000
```

The production-style Gunicorn command is:

```bash
gunicorn --chdir src app:app
```

Gunicorn serves the application locally at:

```text
http://127.0.0.1:8000
```

## Technologies used

- Python
- pandas
- NumPy
- scikit-learn
- Joblib
- Flask
- Gunicorn
- HTML
- CSS
- Render

## External resources

- [Flask documentation](https://flask.palletsprojects.com/)
- [scikit-learn documentation](https://scikit-learn.org/)
- [Gunicorn documentation](https://docs.gunicorn.org/)
- [Render documentation](https://render.com/docs)

## Limitations

This model was developed from a small historical dataset representing a specific population. Its performance has not been externally validated, and the dataset has substantial missingness in some predictors, particularly insulin and skin thickness.

The displayed probability is a model output rather than a validated individualized clinical risk estimate.

## Disclaimer

This application is an educational machine-learning demonstration. It is not a validated clinical tool and must not be used for diagnosis, screening, treatment, or other medical decisions.