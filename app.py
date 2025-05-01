import streamlit as st
import pandas as pd
import pickle

# Load model and preprocessors

with open("CrimeData.joblib", "rb") as f:
    bundle = pickle.load(f)
    return bundle

model = bundle["model1"]
label_encoder = bundle["encoder_y"]
ohe = bundle["encoder_x"]
selector = bundle["selectkbest"]

st.title("🚓 Crime Category Prediction App")
st.markdown("### Enter Crime Incident Details:")

# Manual input widgets
area = st.selectbox("Area", ['N Hollywood',
 'Newton',
 'Mission',
 '77th Street',
 'Northeast',
 'Hollenbeck',
 'Pacific',
 'Van Nuys',
 'Devonshire',
 'Wilshire',
 'Hollywood',
 'Harbor',
 'Topanga',
 'Central',
 'West Valley',
 'Olympic',
 'Foothill',
 'West LA',
 'Southeast',
 'Southwest',
 'Rampart'])
time_occurred = st.number_input("Time Occurred (e.g., 1330 for 1:30 PM)", min_value=0, max_value=2359, step=1)
part_1_2 = st.number_input("Part 1-2", min_value=1, max_value=2, step=1)
victim_age = st.number_input("Victim Age", min_value=0, max_value=100, step=1)
victim_sex = st.selectbox("Victim Sex", ['M', 'F', 'X', 'H'])
victim_descent = st.selectbox("Victim_Descent", ['W',
 'H',
 'B',
 'X',
 'O',
 'A',
 'K',
 'C',
 'F',
 'I',
 'J',
 'Z',
 'V',
 'P',
 'D',
 'U',
 'G'])
status = st.selectbox("Status", ['IC', 'AO', 'AA', 'JA', 'JO'])
weapon_description = st.selectbox("Weapon_Description", ['UNKNOWN WEAPON/OTHER WEAPON',
 'STRONG-ARM (HANDS, FIST, FEET OR BODILY FORCE)',
 'VERBAL THREAT',
 'OTHER KNIFE',
 'HAND GUN',
 'VEHICLE',
 'FIRE',
 'PIPE/METAL PIPE',
 'KNIFE WITH BLADE 6INCHES OR LESS',
 'BLUNT INSTRUMENT',
 'CLUB/BAT',
 'SEMI-AUTOMATIC PISTOL',
 'ROCK/THROWN OBJECT',
 'MACHETE',
 'UNKNOWN FIREARM',
 'AIR PISTOL/REVOLVER/RIFLE/BB GUN',
 'TOY GUN',
 'FIXED OBJECT',
 'UNKNOWN TYPE CUTTING INSTRUMENT',
 'FOLDING KNIFE',
 'HAMMER',
 'PHYSICAL PRESENCE',
 'MACE/PEPPER SPRAY',
 'OTHER CUTTING INSTRUMENT',
 'BOARD',
 'BOTTLE',
 'KITCHEN KNIFE',
 'RIFLE',
 'KNIFE WITH BLADE OVER 6 INCHES IN LENGTH',
 'SCREWDRIVER',
 'STICK',
 'SIMULATED GUN',
 'BELT FLAILING INSTRUMENT/CHAIN',
 'CONCRETE BLOCK/BRICK',
 'AXE',
 'ICE PICK',
 'REVOLVER',
 'OTHER FIREARM',
 'SCISSORS',
 'STARTER PISTOL/REVOLVER',
 'GLASS',
 'SHOTGUN',
 'BRASS KNUCKLES',
 'SWITCH BLADE',
 'TIRE IRON',
 'SAWED OFF RIFLE/SHOTGUN',
 'CAUSTIC CHEMICAL/POISON',
 'SCALDING LIQUID',
 'DEMAND NOTE',
 'BOMB THREAT',
 'BOWIE KNIFE',
 'STUN GUN',
 'MARTIAL ARTS WEAPONS',
 'RAZOR BLADE',
 'HECKLER & KOCH 93 SEMIAUTOMATIC ASSAULT RIFLE',
 'ASSAULT WEAPON/UZI/AK47/ETC',
 'CLEAVER'])

if st.button("🔍 Predict Crime Category"):
    try:
        # Construct input DataFrame
        input_dict = {
            'Area': [area],
            'Time_Occurred': [time_occurred],
            'Part 1-2': [part_1_2],
            'Victim_Age': [victim_age],
            'Victim_Sex': [victim_sex],
            'Victim_Descent': [victim_descent],
            'Status':[status],
            'Weapon Description':[weapon_description]
        }
        input_df = pd.DataFrame(input_dict)

        # Split categorical and numerical
        categorical_cols = ['Area', 'Victim_Sex', 'Victim_Descent', 'Status', 'Weapon Description']
        numerical_cols = ['Time_Occurred','Part 1-2', 'Victim_Age']

        # OHE
        encoded_cat = ohe.transform(input_df[categorical_cols])
        encoded_cat_df = pd.DataFrame(
            encoded_cat,
            columns=ohe.get_feature_names_out(categorical_cols)
        )

        # Combine with numeric
        final_input = pd.concat([encoded_cat_df, input_df[numerical_cols].reset_index(drop=True)], axis=1)

        # Feature selection
        selected_input = selector.transform(final_input)

        # Predict
        y_pred = model.predict(selected_input)
        prediction = label_encoder.inverse_transform(y_pred)

        st.success(f"🕵️ Predicted Crime Category: **{prediction[0]}**")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
