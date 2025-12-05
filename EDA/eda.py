import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Dataset path
data_path = "/home/midstan/Documents/Health Insurance Premium/Dataset/insurance.csv"

# Load the dataset
df = pd.read_csv(data_path)

# Basic EDA: Display dataset info (console output)
print("Dataset Shape:", df.shape)
print("\nDataset Info:")
print(df.info())
print("\nDataset Description:")
print(df.describe())
print("\nFirst 5 Rows:")
print(df.head())

# Create output directory for plots
output_dir = "/home/midstan/Documents/Health Insurance Premium/EDA/plots"
os.makedirs(output_dir, exist_ok=True)

# 1. Age Distribution Histogram – to see age range and spread of insured individuals.
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='age', kde=True, bins=30)
plt.title('Age Distribution of Insured Individuals')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig(os.path.join(output_dir, '1_age_distribution.png'), dpi=300, bbox_inches='tight')
plt.close()  # Close to free memory; use plt.show() if you want inline display

# 2. Average BMI by Gender – to compare body mass trends across males and females.
plt.figure(figsize=(10, 6))
avg_bmi_gender = df.groupby('sex')['bmi'].mean().reset_index()
sns.barplot(data=avg_bmi_gender, x='sex', y='bmi')
plt.title('Average BMI by Gender')
plt.xlabel('Gender')
plt.ylabel('Average BMI')
plt.savefig(os.path.join(output_dir, '2_avg_bmi_by_gender.png'), dpi=300, bbox_inches='tight')
plt.close()

# 3. BMI vs Age (with Gender hue) – to visualize how BMI changes with age for each gender.
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='age', y='bmi', hue='sex', alpha=0.6)
plt.title('BMI vs Age by Gender')
plt.xlabel('Age')
plt.ylabel('BMI')
plt.savefig(os.path.join(output_dir, '3_bmi_vs_age_by_gender.png'), dpi=300, bbox_inches='tight')
plt.close()

# 4. Expenses vs BMI (with Smoker hue) – to show how smoking status impacts premium costs relative to BMI.
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='bmi', y='expenses', hue='smoker', alpha=0.6)
plt.title('Expenses vs BMI by Smoker Status')
plt.xlabel('BMI')
plt.ylabel('Expenses')
plt.savefig(os.path.join(output_dir, '4_expenses_vs_bmi_by_smoker.png'), dpi=300, bbox_inches='tight')
plt.close()

# 5. Expenses vs Age (with Smoker hue) – to highlight the combined effect of age and smoking on expenses.
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='age', y='expenses', hue='smoker', alpha=0.6)
plt.title('Expenses vs Age by Smoker Status')
plt.xlabel('Age')
plt.ylabel('Expenses')
plt.savefig(os.path.join(output_dir, '5_expenses_vs_age_by_smoker.png'), dpi=300, bbox_inches='tight')
plt.close()

# 6. Average Expenses by Number of Children – to examine how dependents affect premiums.
plt.figure(figsize=(10, 6))
avg_expenses_children = df.groupby('children')['expenses'].mean().reset_index()
sns.barplot(data=avg_expenses_children, x='children', y='expenses')
plt.title('Average Expenses by Number of Children')
plt.xlabel('Number of Children')
plt.ylabel('Average Expenses')
plt.savefig(os.path.join(output_dir, '6_avg_expenses_by_children.png'), dpi=300, bbox_inches='tight')
plt.close()

# 7. Count of Male and Female in Each Region – to understand demographic distribution.
plt.figure(figsize=(10, 6))
region_gender_count = df.groupby(['region', 'sex']).size().unstack(fill_value=0)
region_gender_count.plot(kind='bar', stacked=True)
plt.title('Count of Males and Females by Region')
plt.xlabel('Region')
plt.ylabel('Count')
plt.legend(title='Gender')
plt.savefig(os.path.join(output_dir, '7_count_male_female_by_region.png'), dpi=300, bbox_inches='tight')
plt.close()

# 8. Average Expenses by Region – to see regional variation in insurance costs.
plt.figure(figsize=(10, 6))
avg_expenses_region = df.groupby('region')['expenses'].mean().reset_index()
sns.barplot(data=avg_expenses_region, x='region', y='expenses')
plt.title('Average Expenses by Region')
plt.xlabel('Region')
plt.ylabel('Average Expenses')
plt.savefig(os.path.join(output_dir, '8_avg_expenses_by_region.png'), dpi=300, bbox_inches='tight')
plt.close()

# 9. Average BMI by Region – to check health trends across regions.
plt.figure(figsize=(10, 6))
avg_bmi_region = df.groupby('region')['bmi'].mean().reset_index()
sns.barplot(data=avg_bmi_region, x='region', y='bmi')
plt.title('Average BMI by Region')
plt.xlabel('Region')
plt.ylabel('Average BMI')
plt.savefig(os.path.join(output_dir, '9_avg_bmi_by_region.png'), dpi=300, bbox_inches='tight')
plt.close()

# 10. Average Expenses by Smoker Status – to emphasize the strong impact of smoking.
plt.figure(figsize=(10, 6))
avg_expenses_smoker = df.groupby('smoker')['expenses'].mean().reset_index()
sns.barplot(data=avg_expenses_smoker, x='smoker', y='expenses')
plt.title('Average Expenses by Smoker Status')
plt.xlabel('Smoker Status')
plt.ylabel('Average Expenses')
plt.savefig(os.path.join(output_dir, '10_avg_expenses_by_smoker.png'), dpi=300, bbox_inches='tight')
plt.close()

# Optional: Save console tables as images (e.g., describe() table). Uncomment if needed.
# fig, ax = plt.subplots(figsize=(12, 6))
# ax.axis('tight')
# ax.axis('off')
# table = ax.table(cellText=df.describe().round(2).values, colLabels=df.describe().columns, cellLoc='center', loc='center')
# table.auto_set_font_size(False)
# table.set_fontsize(10)
# table.scale(1.2, 1.5)
# plt.title('Dataset Description Table')
# plt.savefig(os.path.join(output_dir, 'dataset_description_table.png'), dpi=300, bbox_inches='tight')
# plt.close()

print(f"\nAll 10 plots saved to: {output_dir}")
print("EDA complete! No errors expected now.")