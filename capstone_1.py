import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
def load_data(file_path):
    return pd.read_csv(file_path, comment='#', header=None).to_numpy()

male = load_data("nhanes_adult_male_bmx_2020.csv")
female = load_data("nhanes_adult_female_bmx_2020.csv")

# Extract weight data
male_weights = male[:, 0]
female_weights = female[:, 0]

# Plot histograms
fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
axes[0].hist(female_weights, bins=30, color='pink', edgecolor='black')
axes[0].set_title("Female Weight Distribution")
axes[1].hist(male_weights, bins=30, color='blue', edgecolor='black')
axes[1].set_title("Male Weight Distribution")
plt.xlabel("Weight (kg)")
plt.show()

# Boxplot comparison
plt.boxplot([female_weights, male_weights], labels=["Female", "Male"])
plt.title("Boxplot of Weights")
plt.ylabel("Weight (kg)")
plt.show()

# Compute basic statistics
def compute_statistics(data):
    return {
        "Mean": np.mean(data),
        "Median": np.median(data),
        "Std Dev": np.std(data),
        "Skewness": pd.Series(data).skew(),
        "Kurtosis": pd.Series(data).kurt()
    }

male_stats = compute_statistics(male_weights)
female_stats = compute_statistics(female_weights)
print("Male Weight Statistics:", male_stats)
print("Female Weight Statistics:", female_stats)

# Compute BMI and add to female dataset
female_bmi = female[:, 0] / ((female[:, 1] / 100) ** 2)
female = np.column_stack((female, female_bmi))

# Standardize female data
zfemale = (female - female.mean(axis=0)) / female.std(axis=0)

# Scatterplot matrix
columns = ["Height", "Weight", "Waist", "Hip", "BMI"]
zfemale_df = pd.DataFrame(zfemale[:, [1, 0, 6, 5, 7]], columns=columns)
sns.pairplot(zfemale_df)
plt.show()

# Compute correlation coefficients
pearson_corr = zfemale_df.corr(method='pearson')
spearman_corr = zfemale_df.corr(method='spearman')
print("Pearson Correlation:\n", pearson_corr)
print("Spearman Correlation:\n", spearman_corr)

# Compute waist ratios
male_waist_to_height = male[:, 6] / male[:, 1]
male_waist_to_hip = male[:, 6] / male[:, 5]
female_waist_to_height = female[:, 6] / female[:, 1]
female_waist_to_hip = female[:, 6] / female[:, 5]

male = np.column_stack((male, male_waist_to_height, male_waist_to_hip))
female = np.column_stack((female, female_waist_to_height, female_waist_to_hip))

# Boxplot for waist ratios
plt.boxplot([
    female[:, -2], male[:, -2], 
    female[:, -1], male[:, -1]
], labels=["Female W/Ht", "Male W/Ht", "Female W/Hp", "Male W/Hp"])
plt.title("Comparison of Waist Ratios")
plt.show()

# Print standardised measurements for extreme BMI values
sorted_bmi_idx = np.argsort(zfemale[:, 7])
lowest_bmi = zfemale[sorted_bmi_idx[:5]]
highest_bmi = zfemale[sorted_bmi_idx[-5:]]
print("Lowest BMI Individuals:\n", lowest_bmi)
print("Highest BMI Individuals:\n", highest_bmi)
