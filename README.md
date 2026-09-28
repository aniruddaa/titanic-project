# Titanic survival model

A small Streamlit dashboard for exploring a logistic regression model trained to predict Titanic survival from passenger age.

## Run the dashboard

```powershell
python -m pip install -r requirements.txt
python -m streamlit run trainmodel.py
```

The app loads `titanicmodel.pkl` and `titanic.csv` from this directory. It includes an interactive age prediction, the model's survival-probability curve, and observed survival rates by age group.

## Retrain the model

```powershell
python titanic.py
```

The training script saves the model to `titanicmodel.pkl`. The current classifier uses age as its only feature; its predictions do not account for other passenger information.