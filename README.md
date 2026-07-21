# nyc-airbnb-price-prediction

An end-to-end machine learning project that predicts whether an Airbnb listing in New York City belongs to the **high-price** or **low-price** category. This project compares a traditional Logistic Regression model with a feedforward Neural Network to evaluate the trade-offs between model interpretability and predictive performance.

## Project Overview

Airbnb pricing depends on a combination of listing characteristics, host information, location, and property features. The objective of this project is to build a binary classification model capable of predicting whether a listing falls into the top 25% of prices ("high") or the remaining 75% ("low").

Following the complete machine learning lifecycle, this project includes data exploration, preprocessing, feature engineering, model training, hyperparameter tuning, neural network development, and model comparison.

## Dataset

The project uses the **NYC Airbnb Listings** dataset containing features such as:

- Neighborhood
- Room type
- Host information
- Accommodation capacity
- Availability
- Number of bedrooms and beds
- Number of reviews
- Listing statistics
- Other property characteristics

The target variable is **price_category**, where:

- **High** → Listing price is at or above the 75th percentile
- **Low** → Listing price is below the 75th percentile

## Machine Learning Pipeline

### Data Preparation
- Explored class distribution and feature relationships through EDA
- Handled missing values using mean imputation
- Added missing-value indicator features
- Removed unstructured text columns
- Applied one-hot encoding to categorical variables
- Standardized numerical features using `StandardScaler`

### Models

#### Logistic Regression
- Built a baseline binary classification model
- Optimized hyperparameters using GridSearchCV
- Evaluated using Accuracy and F1 Score
- Interpreted feature coefficients to identify key pricing drivers

#### Feedforward Neural Network
- Built using TensorFlow/Keras
- Two hidden layers (32 → 16 neurons)
- ReLU activation functions
- Sigmoid output layer
- SGD optimizer (learning rate = 0.01)
- Binary Cross-Entropy loss
- Trained for 100 epochs with validation monitoring

## Results

| Model | Accuracy | F1 Score |
|--------|---------:|---------:|
| Logistic Regression | **82.6%** | **61.8%** |
| Neural Network | **84.6%** | **68.5%** |

The neural network achieved higher performance on both evaluation metrics, particularly improving the F1 score by nearly 7 percentage points. This suggests that the neural network was better at identifying high-priced listings, while Logistic Regression remained valuable due to its interpretability and ability to explain which features influenced predictions.

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow / Keras
- Matplotlib
- Seaborn
- Jupyter Notebook

## Key Takeaways

- Built an end-to-end supervised machine learning pipeline.
- Compared a traditional linear model with a deep learning approach.
- Used hyperparameter tuning to optimize Logistic Regression.
- Designed and trained a feedforward neural network for binary classification.
- Evaluated models using Accuracy and F1 Score.
- Considered fairness and ethical implications of using location-based features for pricing predictions.

## Future Improvements

- Address class imbalance using techniques such as SMOTE or class weighting.
- Experiment with deeper neural network architectures and dropout layers.
- Evaluate ensemble models such as Random Forest or XGBoost.
- Deploy the best-performing model as an interactive web application.

## Author

**Jenvi Patel**
