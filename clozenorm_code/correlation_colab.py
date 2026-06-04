# paste code into colab / adjust to run however to compare correlation between reproduction and original cloze probabilities 
import pandas as pd
from scipy.stats import pearsonr

df = pd.read_csv('final.csv')

print(df.head())

empirical_prob = df['empirical_prob']
normalized_prob = df['normalized_prob']
cloze_prob = df['Cloze_Probability']

corr_empirical_cloze = pearsonr(empirical_prob.dropna(), cloze_prob.loc[empirical_prob.notna()])

adjusted_prob = empirical_prob.where(empirical_prob != 0, normalized_prob)

corr_adjusted_cloze = pearsonr(adjusted_prob.dropna(), cloze_prob.loc[adjusted_prob.notna()])

print(f"Correlation between empirical probability and cloze probability: {corr_empirical_cloze[0]}")
print(f"Correlation between adjusted probability and cloze probability: {corr_adjusted_cloze[0]}")
