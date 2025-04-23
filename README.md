# CSGO Match Prediction

This project develops a machine learning model to predict the outcome of CSGO matches (win, lose, or draw) based on player and match statistics. The model uses a RandomForestClassifier, with data preprocessing, SMOTE for handling class imbalance, and GridSearchCV for hyperparameter tuning.

## Project Structure
- `csgo_prediction.py`: Python script for data loading, preprocessing, model training, and evaluation.
- `csgo.csv`: Dataset containing CSGO match data (e.g., map, match time, kills, result).
- `README.md`: This file, providing project documentation.

## Dataset
The `csgo.csv` file contains match data with the following columns:
- **Features**:
  - `map`: Map played (e.g., de_dust2, de_inferno).
  - `match_time_s`: Match duration in seconds.
  - `ping`: Network latency.
  - `kills`: Number of kills.
  - `assists`: Number of assists.
  - `deaths`: Number of deaths.
  - `mvps`: Number of MVP awards.
  - `hs_percent`: Headshot percentage.
  - `points`: Player points.
- **Target**:
  - `result`: Match outcome (e.g., win, lose, draw).
- **Other columns** (not used): `day`, `month`, `year`, `date`, `wait_time_s`, `team_a_rounds`, `team_b_rounds`.

## Requirements
To run the project, you need Python 3.x and the following libraries:
- pandas
- scikit-learn
- imblearn

Install them using:
```bash
pip install pandas scikit-learn imblearn
