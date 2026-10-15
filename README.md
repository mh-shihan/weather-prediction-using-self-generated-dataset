# Weather Prediction with a Self-Generated Dataset

## Overview
This project is a complete machine learning workflow for predicting weather conditions using synthetic environmental data. It demonstrates how to generate a realistic dataset from scratch, train a classification model, evaluate its performance, tune hyperparameters, and deploy the trained model for real-time prediction.

The project includes:
- a synthetic data generator in `generator.py`
- a trained prediction pipeline in the notebook workflow
- a model-loading script in `predictor.py`
- a dataset file named `weather_dataset.csv`

The target labels are:
- `Rainy`
- `Sunny`
- `Windy`
- `Storm`
- `Normal`

---

## Project Summary
This project was created to showcase an end-to-end machine learning workflow using a custom-generated dataset. A synthetic dataset of 10,000 rows was generated with realistic environmental variables such as temperature, humidity, wind speed, atmospheric pressure, and pollution levels. These values were then used to create rule-based labels for different weather conditions, allowing a classification model to learn the relationship between environmental inputs and weather outcomes.

The model pipeline combines preprocessing and classification in a robust workflow. The project uses a `RandomForestClassifier` because it handles non-linear relationships well, performs strongly on tabular data, and is highly suitable for this type of classification task. Hyperparameter tuning was also applied using `GridSearchCV` to improve model performance and validate the best configuration.

This project is valuable for a resume because it demonstrates practical skills in synthetic data generation, machine learning model training, evaluation, tuning, and deployment. It shows the ability to build a complete predictive system from scratch, which is a strong indicator of analytical thinking, problem-solving, and applied data science capability.

---

## Project Structure

- `generator.py` — creates the synthetic weather dataset
- `weather_dataset.csv` — generated dataset with 10,000 rows
- `weather_predictor.ipynb` — notebook containing data exploration, training, evaluation, and tuning
- `predictor.py` — loads the saved model and predicts weather from user input
- `README.md` — project documentation

---

## Dataset Generation (`generator.py`)

The dataset was generated using `generator.py`, which creates 10,000 synthetic samples with realistic weather-related variables. The script uses a fixed random seed for reproducibility and produces values in a believable range for environmental conditions.

### Key Features
The generated dataset contains the following variables:

| Column | Description |
| --- | --- |
| `PM2.5` | Fine particulate matter concentration |
| `PM10` | Coarse particulate matter concentration |
| `temperature` | Air temperature in degrees Celsius |
| `humidity` | Relative humidity percentage |
| `wind_speed` | Wind speed |
| `atmospheric_pressure` | Atmospheric pressure |
| `historical_pollution` | Past pollution level |
| `weather` | Target class label |

### How the Data Was Created
The script follows a structured process:

1. Random values are generated for weather-related variables using NumPy.
2. Values are clipped to realistic physical ranges to avoid unrealistic readings.
3. Pollution values are created using a relationship between historical pollution and current measurements.
4. Weather condition scores are computed using domain-based logic:
   - `Rainy` is favored by high humidity and lower atmospheric pressure
   - `Sunny` is favored by higher temperature, lower humidity, and higher pressure
   - `Windy` is driven mainly by increased wind speed
   - `Storm` is associated with high wind, high humidity, and low pressure
   - `Normal` represents conditions close to average values
5. The final label is selected by determining which score is highest for each row.
6. The resulting DataFrame is saved as `weather_dataset.csv`.

This approach ensures the dataset is synthetic but still meaningful and suitable for machine learning use.

---

## Dataset Characteristics

The generated dataset contains 10,000 rows and includes both numerical environmental features and a categorical target variable. The weather labels are balanced by the score-based generation logic, resulting in a robust dataset for classification.

The project uses a step-by-step pipeline:
- split the dataset into training and testing sets
- standardize the feature values
- train a machine learning classifier
- evaluate the performance on unseen data
- tune the model using grid search

---

## Machine Learning Workflow

The model-building process in `weather_predictor.ipynb` follows this flow:

1. Load the dataset from `weather_dataset.csv`
2. Inspect the class distribution and the first rows of data
3. Define the feature matrix `X` and target `y`
4. Split the dataset into training and testing sets using stratification
5. Create a pipeline with:
   - `StandardScaler()` for feature scaling
   - `RandomForestClassifier()` for classification
6. Train the model on the training data
7. Generate predictions on the test data
8. Evaluate using accuracy, precision, recall, F1-score, and classification report
9. Improve performance using `GridSearchCV` hyperparameter tuning
10. Save the best-performing model for later use

### Model Selection
The final model uses a `RandomForestClassifier` inside a `Pipeline`, which makes the workflow easy to reproduce and robust for classification tasks.

### Evaluation Metrics
The notebook evaluates the model based on:
- accuracy
- weighted precision
- weighted recall
- weighted F1-score
- classification report per class

These metrics help determine how well the model generalizes to new weather examples.

---

## Hyperparameter Tuning

The project also includes hyperparameter tuning with `GridSearchCV` to identify the best model configuration. The tuning process explores combinations such as:

- `n_estimators`
- `max_features`
- `max_depth`

This search helps improve the model by selecting the parameters that produce the strongest validation performance.

---

## Saving the Trained Model

After training and tuning, the model is serialized using Python's `pickle` module. This allows the trained pipeline to be reused without retraining the model each time.

The saved model can be loaded and used for prediction in the future.

---

## Prediction Script (`predictor.py`)

The file `predictor.py` is designed to load the saved model and predict the weather condition from user-provided environmental values.

### Input Format
The script expects a comma-separated list of values in this order:

`PM2.5, PM10, temperature, humidity, wind_speed, atmospheric_pressure, historical_pollution`

### Example
```python
PM2.5, PM10, temperature, humidity, wind_speed, atmospheric_pressure, historical_pollution
```

Example input:
```text
30, 55, 29, 62, 6, 1015, 48
```

The script reads these values, builds a DataFrame, and passes them into the trained model to predict the weather class.

---

## How to Run the Project

### 1. Generate the dataset
```bash
python generator.py
```
This creates `weather_dataset.csv` with the synthetic data.

### 2. Open and run the notebook
Open `weather_predictor.ipynb` and execute the cells in order.

### 3. Use the prediction script
```bash
python predictor.py
```
Then enter the feature values when prompted.

---

## Tools and Libraries Used

This project uses the following Python libraries:
- `numpy`
- `pandas`
- `scikit-learn`
- `pickle`

These libraries power the dataset generation, model training, evaluation, tuning, and prediction process.

---

## Conclusion

This project is an excellent example of building a realistic classification problem from synthetic data. It combines domain-based data generation with data science workflows such as preprocessing, model training, evaluation, hyperparameter tuning, and prediction deployment.

By generating the data manually, the project demonstrates how synthetic datasets can be used to prototype and learn machine learning pipelines in a controlled and explainable way.

---

## License and Notes

This project is intended for educational and demonstration purposes. The synthetic dataset is generated for training and experimentation and is not based on real-world weather records.
