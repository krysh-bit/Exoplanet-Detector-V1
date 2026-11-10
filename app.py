
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ---------- PAGE CONFIGURATION ----------
st.set_page_config(
    page_title="Exoplanet Detector V1",
    page_icon="🌌",
    layout="wide"
)

# ---------- FEATURES ----------
FEATURES = [
    "koi_period",
    "koi_depth",
    "koi_duration",
    "koi_prad",
    "koi_impact",
    "koi_steff",
    "koi_srad",
    "koi_model_snr"
]

FEATURE_LABELS = {
    "koi_period": "Orbital Period (days)",
    "koi_depth": "Transit Depth (ppm)",
    "koi_duration": "Transit Duration (hours)",
    "koi_prad": "Planet Radius (Earth radii)",
    "koi_impact": "Impact Parameter",
    "koi_steff": "Star Temperature (K)",
    "koi_srad": "Star Radius (Solar radii)",
    "koi_model_snr": "Transit Signal-to-Noise Ratio"
}

CLASS_COLORS = {
    "CONFIRMED": "#34d399",
    "CANDIDATE": "#fbbf24",
    "FALSE POSITIVE": "#f87171"
}

# ---------- LOAD MODEL AND DATA ----------
@st.cache_resource
def load_model():
    return joblib.load("exoplanet_detector.pkl")


@st.cache_data
def load_dataset():
    data = pd.read_csv("dataset.csv")

    required = FEATURES + ["koi_disposition"]
    missing = [column for column in required if column not in data.columns]

    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}"
        )

    return data


model = load_model()

try:
    df = load_dataset()
    data_error = None
except Exception as error:
    df = None
    data_error = str(error)


# ---------- CUSTOM DESIGN ----------
st.markdown("""
<style>
.stApp {
    background:
        radial-gradient(1px 1px at 10% 15%, #ffffffaa 50%, transparent 100%),
        radial-gradient(1px 1px at 25% 75%, #ffffff88 50%, transparent 100%),
        radial-gradient(2px 2px at 80% 20%, #ffffff99 50%, transparent 100%),
        radial-gradient(1px 1px at 65% 60%, #ffffff88 50%, transparent 100%),
        radial-gradient(1px 1px at 90% 85%, #ffffff77 50%, transparent 100%),
        radial-gradient(2px 2px at 45% 35%, #ffffff66 50%, transparent 100%),
        radial-gradient(ellipse at top, #172554 0%, #080d1d 55%, #050816 100%);
    background-attachment: fixed;
    color: #f1f5f9;
}

.block-container {
    max-width: 1150px;
    padding-top: 2rem;
}

h1, h2, h3, p, label {
    color: #f1f5f9 !important;
}

[data-testid="stSidebar"] {
    background-color: #0b1225 !important;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #ffffff !important;
}

[data-testid="stSidebar"] [role="radiogroup"] label {
    color: #ffffff !important;
    font-weight: 700 !important;
    opacity: 1 !important;
}

[data-testid="stSidebar"] [role="radiogroup"] label p {
    color: #ffffff !important;
    font-size: 16px !important;
    font-weight: 700 !important;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #e2e8f0 !important;
}

[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
    background-color: #26365c !important;
    border-radius: 8px;
}

.hero {
    text-align: center;
    padding: 35px 20px;
    border: 1px solid #334155;
    border-radius: 20px;
    background: linear-gradient(135deg, #172554dd, #312e81cc);
    margin-bottom: 30px;
}

.hero h1 {
    font-size: 40px;
}

.hero p {
    color: #cbd5e1 !important;
}

.result {
    text-align: center;
    padding: 25px;
    border-radius: 16px;
    background: linear-gradient(135deg, #1e3a5f, #312e81);
    border: 1px solid #818cf8;
    margin-top: 25px;
}

.result h2 {
    font-size: 30px;
}

div.stButton > button {
    width: 100%;
    border-radius: 12px;
    background: linear-gradient(90deg, #6366f1, #8b5cf6);
    color: white;
    border: none;
    padding: 12px;
    font-size: 18px;
    font-weight: bold;
}

div.stButton > button:hover {
    color: white;
    border: 1px solid #a5b4fc;
}

.info-card {
    background: #111b33dd;
    border: 1px solid #334155;
    border-radius: 14px;
    padding: 18px;
    margin-bottom: 12px;
}

[data-testid="stMetric"] {
    background: #111b33dd;
    border: 1px solid #334155;
    padding: 14px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)


# ---------- SIDEBAR NAVIGATION ----------
st.sidebar.title("🌌 Exoplanet Detector")
st.sidebar.caption("Kepler Mission • ML Project")

page = st.sidebar.radio(
    "Navigation",
    ["Detector", "About the Project"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.caption("Version 1.0 | Random Forest Classifier")


# ==================================================
# DETECTOR PAGE
# ==================================================

if page == "Detector":

    st.markdown("""
    <div class="hero">
        <h1>🌌 EXOPLANET DETECTOR</h1>
        <p>Exploring distant worlds through machine learning</p>
        <p>KEPLER MISSION • RANDOM FOREST CLASSIFIER • V1.0</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## 🪐 Enter Planetary Measurements")

    st.write(
        "Enter the measurements below to classify a Kepler object."
    )

    with st.container(border=True):

        col1, col2 = st.columns(2)

        with col1:

            koi_period = st.number_input(
                "Orbital Period (days)",
                value=10.0,
                min_value=0.0
            )

            koi_depth = st.number_input(
                "Transit Depth (ppm)",
                value=400.0,
                min_value=0.0
            )

            koi_duration = st.number_input(
                "Transit Duration (hours)",
                value=3.8,
                min_value=0.0
            )

            koi_prad = st.number_input(
                "Planet Radius (Earth radii)",
                value=2.4,
                min_value=0.0
            )

        with col2:

            koi_impact = st.number_input(
                "Impact Parameter",
                value=0.5,
                min_value=0.0
            )

            koi_steff = st.number_input(
                "Star Temperature (K)",
                value=5500.0,
                min_value=0.0
            )

            koi_srad = st.number_input(
                "Star Radius (Solar radii)",
                value=1.0,
                min_value=0.0
            )

            koi_model_snr = st.number_input(
                "Transit Signal-to-Noise Ratio",
                value=10.0,
                min_value=0.0
            )

        detect = st.button("🚀 DETECT EXOPLANET")


    # ---------- PREDICTION ----------
    if detect:

        input_data = pd.DataFrame([{
            "koi_period": koi_period,
            "koi_depth": koi_depth,
            "koi_duration": koi_duration,
            "koi_prad": koi_prad,
            "koi_impact": koi_impact,
            "koi_steff": koi_steff,
            "koi_srad": koi_srad,
            "koi_model_snr": koi_model_snr
        }])

        st.session_state["latest_input"] = input_data

        st.session_state["latest_prediction"] = (
            model.predict(input_data)[0]
        )

        st.session_state["latest_probabilities"] = (
            model.predict_proba(input_data)[0]
        )


    # ---------- DISPLAY RESULTS ----------
    if "latest_prediction" in st.session_state:

        input_data = st.session_state["latest_input"]
        prediction = st.session_state["latest_prediction"]
        probabilities = st.session_state["latest_probabilities"]

        descriptions = {
            "CONFIRMED":
                "The model classifies this object as a confirmed exoplanet.",

            "CANDIDATE":
                "The model classifies this object as an exoplanet candidate that needs further verification.",

            "FALSE POSITIVE":
                "The model classifies this object as a likely false positive."
        }

        st.markdown(f"""
        <div class="result">
            <h2>🔭 Detection Result</h2>
            <h2>{prediction}</h2>
            <p>{descriptions.get(prediction, "Classification completed.")}</p>
        </div>
        """, unsafe_allow_html=True)


        # ---------- PREDICTION PROBABILITIES ----------
        st.markdown("## Prediction Probabilities")

        st.write(
            "These values show the probability estimates produced "
            "by the model for each of its three classes."
        )

        probability_df = pd.DataFrame({
            "Classification": model.classes_,
            "Estimated probability": probabilities
        }).set_index("Classification")

        st.bar_chart(
            probability_df,
            y="Estimated probability",
            color="#8b7cf6"
        )

        st.caption(
            "Model probability estimates are not guarantees or "
            "independent astronomical confirmations."
        )


        # ---------- FEATURE IMPORTANCE ----------
        st.markdown("---")
        st.markdown("## Model Feature Importance")

        st.write(
            "This chart shows the overall feature importance learned "
            "by the Random Forest. It describes the model's use of "
            "features, not the direction or cause of an individual prediction."
        )

        try:

            forest = model.named_steps["model"]

            importance_df = pd.DataFrame({
                "Feature": [
                    FEATURE_LABELS[name] for name in FEATURES
                ],
                "Importance": forest.feature_importances_
            }).sort_values(
                "Importance",
                ascending=True
            ).set_index("Feature")

            # Create feature importance chart
            fig, ax = plt.subplots(figsize=(8, 5))

            ax.barh(
                importance_df.index,
                importance_df["Importance"],
                color="#38bdf8"
            )

            ax.set_xlabel("Importance")
            ax.set_title("Random Forest Feature Importance")
            ax.invert_yaxis()

            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        except (AttributeError, KeyError, TypeError):

            st.info(
                "Feature importance is unavailable for the loaded model."
            )


        # ---------- INPUT VS REAL DATA ----------
        st.markdown("---")
        st.markdown("## Your Measurements Compared with Kepler Data")

        if df is None:

            st.warning(
                "The dataset could not be loaded. Check that dataset.csv "
                "is in the same folder as app.py. Details: "
                + str(data_error)
            )

        else:

            st.write(
                "Each chart shows the distribution of a measurement "
                "in the original Kepler dataset, separated by the three "
                "known classes. The vertical line marks the value you entered."
            )

            chart_features = [
                "koi_prad",
                "koi_period",
                "koi_depth",
                "koi_duration",
                "koi_impact",
                "koi_steff",
                "koi_srad",
                "koi_model_snr"
            ]

            for start in range(0, len(chart_features), 2):

                chart_cols = st.columns(2)

                for offset, feature in enumerate(
                    chart_features[start:start + 2]
                ):

                    with chart_cols[offset]:

                        fig, ax = plt.subplots(figsize=(6, 3.5))

                        feature_data = df[
                            [feature, "koi_disposition"]
                        ].dropna()

                        # Exclude extreme values from the plotted
                        # histogram range for readability.
                        lower = feature_data[feature].quantile(0.01)
                        upper = feature_data[feature].quantile(0.99)

                        plot_data = feature_data[
                            feature_data[feature].between(lower, upper)
                        ]

                        for class_name in [
                            "CONFIRMED",
                            "CANDIDATE",
                            "FALSE POSITIVE"
                        ]:

                            values = plot_data.loc[
                                plot_data["koi_disposition"] == class_name,
                                feature
                            ]

                            ax.hist(
                                values,
                                bins=25,
                                alpha=0.42,
                                density=True,
                                label=class_name,
                                color=CLASS_COLORS[class_name]
                            )

                        entered_value = float(
                            input_data.iloc[0][feature]
                        )

                        ax.axvline(
                            entered_value,
                            color="white",
                            linestyle="--",
                            linewidth=2,
                            label="Your input"
                        )

                        ax.set_title(
                            FEATURE_LABELS[feature],
                            color="white"
                        )

                        ax.set_xlabel("Value", color="white")
                        ax.set_ylabel("Density", color="white")

                        ax.tick_params(colors="white")

                        ax.set_facecolor("#111b33")
                        fig.patch.set_facecolor("#111b33")

                        ax.spines["bottom"].set_color("#94a3b8")
                        ax.spines["left"].set_color("#94a3b8")

                        ax.spines["top"].set_visible(False)
                        ax.spines["right"].set_visible(False)

                        ax.legend(
                            fontsize=7,
                            facecolor="#172554",
                            labelcolor="white"
                        )

                        fig.tight_layout()

                        st.pyplot(
                            fig,
                            use_container_width=True
                        )

                        plt.close(fig)

            st.caption(
                "For readability, each histogram excludes values below "
                "the 1st percentile and above the 99th percentile of that "
                "feature. Your entered value is still marked even if it "
                "falls outside the displayed range."
            )


    st.markdown("---")

    st.caption(
        "Predictions are model classifications, "
        "not independent astronomical confirmations."
    )


# ==================================================
# ABOUT THE PROJECT PAGE
# ==================================================

elif page == "About the Project":

    st.markdown("""
    <div class="hero">
        <h1>🛰️ ABOUT THE PROJECT</h1>
        <p>Understanding the science and machine learning behind the detector</p>
    </div>
    """, unsafe_allow_html=True)


    # About exoplanets
    st.markdown("## 🌠 What is an Exoplanet?")

    st.write("""
    An exoplanet is a planet that exists outside our Solar System
    and orbits another star.

    NASA's Kepler mission observed stars to identify possible planets
    by detecting tiny dips in their brightness when a planet passes
    in front of its star. This is known as the transit method.
    """)


    # How it works
    st.markdown("## ⚙️ How Does This Detector Work?")

    st.markdown("""
    <div class="info-card">
        <h3>01 · Data Collection</h3>
        <p>The project uses the Kepler Exoplanet Search Results dataset.</p>
    </div>

    <div class="info-card">
        <h3>02 · Data Preprocessing</h3>
        <p>The selected measurements are prepared, missing values are handled,
        and the data is divided into training and testing sets.</p>
    </div>

    <div class="info-card">
        <h3>03 · Model Training</h3>
        <p>A Random Forest classifier learns patterns from labelled
        Kepler observations.</p>
    </div>

    <div class="info-card">
        <h3>04 · Prediction</h3>
        <p>The trained model receives eight measurements and predicts
        one of three classifications.</p>
    </div>
    """, unsafe_allow_html=True)


    # Features
    st.markdown("## 🔬 Features Used")

    features = [
        (
            "Orbital Period",
            "Time taken by the object to complete one orbit."
        ),
        (
            "Transit Depth",
            "Amount of decrease in the star's brightness."
        ),
        (
            "Transit Duration",
            "How long the transit event lasts."
        ),
        (
            "Planet Radius",
            "Estimated radius of the planet."
        ),
        (
            "Impact Parameter",
            "Describes the projected transit path across the star."
        ),
        (
            "Star Temperature",
            "Effective temperature of the host star."
        ),
        (
            "Star Radius",
            "Estimated radius of the host star."
        ),
        (
            "Transit Signal-to-Noise Ratio",
            "Strength of the detected transit signal relative to noise."
        )
    ]

    for name, description in features:

        with st.expander(name):
            st.write(description)


    # Prediction classes
    st.markdown("## 🎯 Prediction Classes")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🟢 CONFIRMED")
        st.write(
            "Classified as a confirmed exoplanet in the dataset."
        )

    with col2:
        st.markdown("### 🟡 CANDIDATE")
        st.write(
            "A possible exoplanet requiring further verification."
        )

    with col3:
        st.markdown("### 🔴 FALSE POSITIVE")
        st.write(
            "An observed signal classified as something "
            "other than a genuine planet."
        )


    # Model information
    st.markdown("## 🧠 Machine Learning Model")

    st.write("""
    **Algorithm:** Random Forest Classifier

    **Task:** Supervised Multiclass Classification

    **Dataset:** Kepler Exoplanet Search Results

    **Model Evaluation:** Stratified Cross-Validation, Macro F1-Score,
    Accuracy, Precision, Recall and Confusion Matrix.
    """)


    # Creator section
    st.markdown("## 👨‍🚀 Created By")

    st.markdown("""
    <div class="info-card">
        <h2>KRISH SHARMA</h2>
        <p>Machine Learning & Astronomy Enthusiast</p>
        <p>Project: Exoplanet Detector V1</p>
        <p>Built with Python, Pandas, Scikit-learn and Streamlit.</p>
        <p>Upgraded versions under construction.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.caption(
        "Exoplanet Detector V1 • "
        "Exploring the universe, one prediction at a time."
    )
