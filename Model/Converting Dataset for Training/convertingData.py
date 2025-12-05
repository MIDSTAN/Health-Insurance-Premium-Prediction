# ============================================================
# 📄 File: convertingData.py
# 📂 Location: /home/midstan/Documents/Health Insurance Premium/Model/Converting Dataset for Training/
# ============================================================

import pandas as pd
import os

# ------------------------------------------------------------
# 📥 Step 1: Load the original dataset
# ------------------------------------------------------------
input_path = "/home/midstan/Documents/Health Insurance Premium/Dataset/insurance.csv"
df = pd.read_csv(input_path)

# ------------------------------------------------------------
# 🧭 Step 2: Define mapping dictionaries
# ------------------------------------------------------------
sex_map = {'male': 1, 'female': 0}
smoker_map = {'yes': 1, 'no': 0}
region_map = {'northeast': 1, 'northwest': 2, 'southeast': 3, 'southwest': 4}

# ------------------------------------------------------------
# 🔄 Step 3: Apply the mappings
# ------------------------------------------------------------
df['sex'] = df['sex'].map(sex_map)
df['smoker'] = df['smoker'].map(smoker_map)
df['region'] = df['region'].map(region_map)

# ------------------------------------------------------------
# 💾 Step 4: Save the converted dataset
# ------------------------------------------------------------
output_dir = "/home/midstan/Documents/Health Insurance Premium/Model/Converting Dataset for Training/Dataset"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "insurance_converted.csv")

df.to_csv(output_path, index=False)

# ------------------------------------------------------------
# ✅ Step 5: Confirmation message
# ------------------------------------------------------------
print(f"✅ Dataset converted successfully and saved at:\n{output_path}")
