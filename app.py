import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved model and preprocessing objects
with open("model_bundle.pkl", "rb") as f:
        bundle = pickle.load(f)
        model = bundle["model"]
        y_encoder = bundle["encoder_Y"]
        x_encoder = bundle["encoder_X"]
        selected_features = bundle["selector"]
            

# Streamlit UI
st.title("Crime Category Prediction App")
st.markdown("Predict crime categories based on input details.")

# Input fields
area_name = st.selectbox("Area Name", ['N Hollywood', 'Newton', 'Mission', '77th Street', 'Northeast',
       'Hollenbeck', 'Pacific', 'Van Nuys', 'Devonshire', 'Wilshire',
       'Hollywood', 'Harbor', 'Topanga', 'Central', 'West Valley',
       'Olympic', 'Foothill', 'West LA', 'Southeast', 'Southwest',
       'Rampart'])
part = st.number_input("Part_1_2", min_value=1, max_value=2)
status = st.selectbox("Status", ['IC', 'AO', 'AA', 'JA', 'JO'])
time_occurred = st.number_input("Time Occurred (0000 to 2359)", min_value=0, max_value=2359)
victim_age = st.number_input("Victim Age", min_value=0, max_value=100)
victim_descent = st.selectbox("Victim Descent", ['W', 'H', 'B', 'X', 'O', 'A', 'K', 'C', 'F', 'I', 'J', 'Z', 'V',
       'P', 'D', 'U', 'G'])
victim_sex = st.selectbox("Victim Sex", ["M", "F", "X", "H"])
weapon_desc = st.selectbox("Weapon Description", ['UNKNOWN WEAPON/OTHER WEAPON',
       'STRONG-ARM (HANDS, FIST, FEET OR BODILY FORCE)', 'VERBAL THREAT',
       'OTHER KNIFE', 'HAND GUN', 'VEHICLE', 'FIRE', 'PIPE/METAL PIPE',
       'KNIFE WITH BLADE 6INCHES OR LESS', 'BLUNT INSTRUMENT', 'CLUB/BAT',
       'SEMI-AUTOMATIC PISTOL', 'ROCK/THROWN OBJECT', 'MACHETE',
       'UNKNOWN FIREARM', 'AIR PISTOL/REVOLVER/RIFLE/BB GUN', 'TOY GUN',
       'FIXED OBJECT', 'UNKNOWN TYPE CUTTING INSTRUMENT', 'FOLDING KNIFE',
       'HAMMER', 'PHYSICAL PRESENCE', 'MACE/PEPPER SPRAY',
       'OTHER CUTTING INSTRUMENT', 'BOARD', 'BOTTLE', 'KITCHEN KNIFE',
       'RIFLE', 'KNIFE WITH BLADE OVER 6 INCHES IN LENGTH', 'SCREWDRIVER',
       'STICK', 'SIMULATED GUN', 'BELT FLAILING INSTRUMENT/CHAIN',
       'CONCRETE BLOCK/BRICK', 'AXE', 'ICE PICK', 'REVOLVER',
       'OTHER FIREARM', 'SCISSORS', 'STARTER PISTOL/REVOLVER', 'GLASS',
       'SHOTGUN', 'BRASS KNUCKLES', 'SWITCH BLADE', 'TIRE IRON',
       'SAWED OFF RIFLE/SHOTGUN', 'CAUSTIC CHEMICAL/POISON',
       'SCALDING LIQUID', 'DEMAND NOTE', 'BOMB THREAT', 'BOWIE KNIFE',
       'STUN GUN', 'MARTIAL ARTS WEAPONS', 'RAZOR BLADE',
       'HECKLER & KOCH 93 SEMIAUTOMATIC ASSAULT RIFLE',
       'ASSAULT WEAPON/UZI/AK47/ETC', 'CLEAVER'])

# Prepare input
input_df = pd.DataFrame({
    'Area_Name': [area_name],
    'Part_1_2': [part],
    'Status': [status],
    'Time_Occurred': [time_occurred],
    'Victim_Age': [victim_age],
    'Victim_Descent': [victim_descent],
    'Victim_Sex': [victim_sex],
    'Weapon_Description': [weapon_desc]
})

# Predict button
if st.button("Predict Crime Category"):
            try:
                        categorical_data = {
                'Area_Name': area_name,
                'Status': status,
                'Victim_Descent': victim_descent,
                'Victim_Sex': victim_sex,
                'Weapon_Description': weapon_desc
            }

            numeric_data = {
                'Part 1 or 2': part,
                'Time_occurred': time_occurred,
                'Victim_age': victim_age
            }
    # Encode features
    encoded_input = x_encoder.transform(categorical_data)
    
    # Feature selection
    selected_features = selector.transform(encoded_input)
    
    # Make prediction
    prediction = model.predict(selected_features)
    prediction_label = y_encoder.inverse_transform(prediction)

    st.subheader("Prediction Result")
    st.write(f"**Predicted Crime Category:** {prediction_label[0]}")
