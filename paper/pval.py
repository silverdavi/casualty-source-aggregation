import scipy.stats as stats
import math

# Null hypothesis: uniform random bombing
p_null = 0.733 # population share of women and children

# OHCHR sample
n_ohchr = 8119
obs_ohchr = int(8119 * 0.693)
z_ohchr = (obs_ohchr - n_ohchr * p_null) / math.sqrt(n_ohchr * p_null * (1 - p_null))
p_ohchr = stats.norm.cdf(z_ohchr)

# MoH sample (assuming ~35,000 fully identified for the demographic breakdown)
n_moh = 35000
obs_moh = int(35000 * 0.560)
z_moh = (obs_moh - n_moh * p_null) / math.sqrt(n_moh * p_null * (1 - p_null))
p_moh = stats.norm.cdf(z_moh)

print(f"OHCHR Z-score: {z_ohchr:.2f}, p-value: {p_ohchr:.2e}")
print(f"MoH Z-score: {z_moh:.2f}, p-value: {p_moh:.2e}")
