# KEPLER
# KEPLER Planet Classification

A machine learning project to classify planets discovered by KEPLER as Candidate, False Positive, or Confirmed based on the `koi_disposition` column.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)

## Installation

### Prerequisites
- Python 3.x
- Required libraries listed in `requirements.txt`

### Steps
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/kepler-planet-classification.git
    ```
2. Navigate to the project directory:
    ```bash
    cd kepler-planet-classification
    ```
3. Create a virtual environment:
    ```bash
    python3 -m venv venv
    ```
4. Activate the virtual environment:
    ```bash
    source venv/bin/activate
    ```
5. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Running the Project
1. Download the KEPLER data from [this link](https://drive.google.com/drive/folders/1GwqC4STc_KgVPofacQUzKHBMHQsmflvY?usp=sharing) and place the CSV file in the project directory.
2. Run the script to train and evaluate the model:
    ```bash
    python classify_planets.py
    ```

### Examples
```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV

# Load the data into a Pandas DataFrame
url = 'path_to_your_downloaded_kepler_data.csv'
data = pd.read_csv(url)

# Display the first few rows of the dataset
print(data.head())

# Check for missing values
print(data.isnull().sum())

# Drop rows with missing values for simplicity
data = data.dropna()

# Encode categorical columns if necessary (e.g., koi_disposition)
label_encoder = LabelEncoder()
data['koi_disposition'] = label_encoder.fit_transform(data['koi_disposition'])

# Check the class distribution
print(data['koi_disposition'].value_counts())

# Define features (X) and the target (y)
features = data.drop(columns=['koi_disposition'])  # Drop the target column
target = data['koi_disposition']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Initialize the model
model = RandomForestClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Define a parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Get the best parameters and model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Evaluate the best model
y_pred_best = best_model.predict(X_test)
print("Best Model Accuracy:", accuracy_score(y_test, y_pred_best))
print("Best Model Classification Report:\n", classification_report(y_test, y_pred_best, target_names=label_encoder.classes_))
