from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent


@st.cache_resource
def load_model():
    return joblib.load(BASE_DIR / "titanicmodel.pkl")


@st.cache_data
def load_data():
    return pd.read_csv(BASE_DIR / "titanic.csv")


st.set_page_config(
    page_title="Titanic survival model",
    page_icon=":material/directions_boat:",
    layout="wide",
)

model = load_model()
passengers = load_data()
survival_class_index = list(model.classes_).index(1)
ages = passengers["Age"].dropna()
age_min = int(ages.min())
age_max = int(ages.max())

st.title("Titanic survival model")
st.caption("Logistic regression · trained on passenger age only")

with st.container(horizontal=True):
    st.metric("Passengers in dataset", f"{len(passengers):,}")
    st.metric("Survival rate", f"{passengers['Survived'].mean():.1%}")
    st.metric("Age values available", f"{passengers['Age'].notna().sum():,}")

st.subheader("Explore a prediction")
prediction_column, curve_column = st.columns([1, 2])

with prediction_column:
    age = st.slider("Passenger age", min_value=age_min, max_value=age_max, value=30)
    passenger = pd.DataFrame({"Age": [age]})
    survival_probability = model.predict_proba(passenger)[0, survival_class_index]
    prediction = model.predict(passenger)[0]
    st.metric("Predicted survival probability", f"{survival_probability:.1%}")
    st.progress(float(survival_probability))
    if prediction == 1:
        st.success("Model prediction: survived")
    else:
        st.warning("Model prediction: did not survive")

with curve_column:
    st.markdown("**Model response by age**")
    age_range = pd.DataFrame({"Age": range(age_min, age_max + 1)})
    age_range["Survival probability"] = model.predict_proba(age_range)[
        :, survival_class_index
    ]
    probability_chart = (
        alt.Chart(age_range)
        .mark_line(color="#087e8b", strokeWidth=3)
        .encode(
            x=alt.X("Age:Q", title="Age (years)"),
            y=alt.Y(
                "Survival probability:Q",
                title="Survival probability",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%"),
            ),
            tooltip=[
                alt.Tooltip("Age:Q", title="Age", format=".0f"),
                alt.Tooltip(
                    "Survival probability:Q", title="Survival probability", format=".1%"
                ),
            ],
        )
    )
    st.altair_chart(probability_chart)

st.subheader("What happened in the dataset")
observed = passengers[["Age", "Survived"]].dropna().copy()
observed["Age group"] = pd.cut(
    observed["Age"],
    bins=[0, 12, 18, 30, 45, 60, age_max + 1],
    labels=["0-11", "12-17", "18-29", "30-44", "45-59", "60+"],
    right=False,
)
age_group_rates = (
    observed.groupby("Age group", observed=True)["Survived"]
    .mean()
    .rename("Observed survival rate")
    .reset_index()
)
st.bar_chart(age_group_rates, x="Age group", y="Observed survival rate")

with st.expander("Preview passenger records"):
    st.dataframe(passengers.head(20), hide_index=True)

st.caption(
    "This model uses age alone. The age-group chart shows observed outcomes and "
    "does not imply that age caused them."
)