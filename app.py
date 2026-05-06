import streamlit as st
import pandas as pd
import pickle


# page config
st.set_page_config(
    # Title and icon for the browser
    page_title="Perfume recomondation",
    page_icon="🪷",
    # Make the content take up the width of the page
    layout="wide",
)
st.write("""
# Perfume recomndiation 

This app predicts your perfect **Perfume**!
""")

# Load model + encoders
model = pickle.load(open("perfume_model.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
 # load dataset
df = pd.read_csv("cperfume_dataset.csv")
# title
st.write("## Find your perfect scent!")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 15, 60, 25)
   

with col2:
    gender = st.selectbox("Gender", encoders["Gender"].classes_)
    weather = st.selectbox("Weather", encoders["Weather"].classes_)
    occasion = st.selectbox("Occasion", encoders["Occasion"].classes_)
    strength = st.selectbox("Strength", encoders["Strength"].classes_)

#predection button

if st.button("Get Recommendation"):
    # scale age
    age_scaled = scaler.transform([[age]])[0][0]
    # create input data
    input_data = pd.DataFrame([[
        age_scaled,
        encoders["Gender"].transform([gender])[0],
        encoders["Weather"].transform([weather])[0],
        encoders["Occasion"].transform([occasion])[0],
        encoders["Strength"].transform([strength])[0]
    ]], columns=["Age","Gender","Weather","Occasion","Strength"])


    # Predict
    pred = model.predict(input_data)[0]
    # get probabilities
    probs = model.predict_proba(input_data)[0]
    # decode output
    result = encoders["Preferred_Scent"].inverse_transform([pred])[0]
    # create dataframe for plots
    labels = encoders["Preferred_Scent"].classes_

    prob_df = pd.DataFrame({
        "Scent": labels,
        "Probability": probs
    }).sort_values(by="Probability", ascending=False)
    # show result
    st.success(f"Recommended Scent: {result}")

  
    # scatter plot
    # use full dataset (not filtered)
    scatter_df = df[["Age", "Preferred_Scent", "Gender"]].dropna().copy()

    # convert scent to numeric
    scatter_df["Scent_code"] = scatter_df["Preferred_Scent"].astype("category").cat.codes

    # plot
    st.write("## People with similar preferences")
  
# convert scent to numeric for plotting

    st.scatter_chart(
        scatter_df,
        x="Age",
        y="Preferred_Scent",
        color="Gender"
    )



    # similar scents
    st.write("""
             ----
    ## You may also like

    """)

    similar = prob_df.iloc[1:4]

    cols = st.columns(3)
    for i, (_, row) in enumerate(similar.iterrows()):
        with cols[i]:
            st.write(f"""### 
                     {row['Scent']}
""")
            st.write(f"""
                     Match: {row['Probability']:.2%}
""")



    # feature importance
    st.write("## Why this recommendation?")

    importance = model.feature_importances_

    feat_df = pd.DataFrame({
        "Feature": ["Age","Gender","Weather","Occasion","Strength"],
        "Importance": importance
    }).sort_values(by="Importance", ascending=False)

    st.area_chart(feat_df.set_index("Feature"))