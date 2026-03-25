import numpy as np
import matplotlib.pyplot as plt

# Problem 3: PCA - Best Places to Live

factor_names = ['Climate', 'Housing', 'Healthcare', 'Crime',
                'Transportation', 'Education', 'Arts', 'Recreation', 'Economy']

# --- Part 1: Read the data ---
city_names = []
ratings = []

with open('places.txt', 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split()

        # Find where the numeric data starts
        name_parts = []
        nums = []
        for p in parts:
            try:
                val = float(p)
                nums.append(val)
            except ValueError:
                if len(nums) == 0:
                    name_parts.append(p)

        city_names.append(' '.join(name_parts))
        ratings.append(nums[:9])  # First 9 columns are the ratings

ratings = np.array(ratings)
print(f"Data shape: {ratings.shape} ({ratings.shape[0]} cities, {ratings.shape[1]} factors)")

# --- Part 2: Log10 transformation ---
X_log = np.log10(ratings)

# --- Part 3: PCA on log-transformed data ---
mu_log = np.mean(X_log, axis=0)
X_centered_log = X_log - mu_log
U_log, S_log, Vt_log = np.linalg.svd(X_centered_log, full_matrices=False)

# --- Part 4: First two principal components and interpretation ---
v1_log = Vt_log[0, :]
v2_log = Vt_log[1, :]

print("\n--- Log-transformed PCA ---")
print(f"\nPC1 = {np.array2string(v1_log, precision=4, suppress_small=True)}")
print("PC1 interpretation:")
for i in np.argsort(np.abs(v1_log))[::-1]:
    print(f"  {factor_names[i]:>16}: {v1_log[i]:+.4f}")

print(f"\nPC2 = {np.array2string(v2_log, precision=4, suppress_small=True)}")
print("PC2 interpretation:")
for i in np.argsort(np.abs(v2_log))[::-1]:
    print(f"  {factor_names[i]:>16}: {v2_log[i]:+.4f}")

# Variance explained
var_explained_log = S_log ** 2 / np.sum(S_log ** 2)
print(f"\nVariance explained: PC1={var_explained_log[0]:.2%}, PC2={var_explained_log[1]:.2%}, "
      f"Total={var_explained_log[:2].sum():.2%}")

# --- Part 5: Project and scatter plot ---
scores_log = X_centered_log @ Vt_log[:2, :].T  # Equivalent to U[:,:2] * S[:2]

plt.figure(figsize=(12, 8))
plt.scatter(scores_log[:, 0], scores_log[:, 1], alpha=0.5, s=20)

# Identify outliers (distance > mean + 2*std)
dists = np.sqrt(scores_log[:, 0] ** 2 + scores_log[:, 1] ** 2)
threshold = np.mean(dists) + 2 * np.std(dists)
outlier_idx = np.where(dists > threshold)[0]

print("\nOutlier cities (log-transformed):")
for i in outlier_idx:
    plt.annotate(city_names[i], (scores_log[i, 0], scores_log[i, 1]), fontsize=7)
    print(f"  {city_names[i]}: PC1={scores_log[i,0]:.3f}, PC2={scores_log[i,1]:.3f}")

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA Scatter Plot (Log-transformed)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('p3_log_scatter.png', dpi=150, bbox_inches='tight')
plt.show()

# --- Part 6: Repeat with Z-score normalization ---

print("\n" + "=" * 60)
print("--- Z-score normalization ---")

mu_z = np.mean(ratings, axis=0)
std_z = np.std(ratings, axis=0, ddof=0)
X_zscore = (ratings - mu_z) / std_z

mu_zs = np.mean(X_zscore, axis=0)
X_centered_z = X_zscore - mu_zs  # Already zero-mean, but being explicit
U_z, S_z, Vt_z = np.linalg.svd(X_centered_z, full_matrices=False)

v1_z = Vt_z[0, :]
v2_z = Vt_z[1, :]

print(f"\nPC1 = {np.array2string(v1_z, precision=4, suppress_small=True)}")
print("PC1 interpretation:")
for i in np.argsort(np.abs(v1_z))[::-1]:
    print(f"  {factor_names[i]:>16}: {v1_z[i]:+.4f}")

print(f"\nPC2 = {np.array2string(v2_z, precision=4, suppress_small=True)}")
print("PC2 interpretation:")
for i in np.argsort(np.abs(v2_z))[::-1]:
    print(f"  {factor_names[i]:>16}: {v2_z[i]:+.4f}")

var_explained_z = S_z ** 2 / np.sum(S_z ** 2)
print(f"\nVariance explained: PC1={var_explained_z[0]:.2%}, PC2={var_explained_z[1]:.2%}, "
      f"Total={var_explained_z[:2].sum():.2%}")

scores_z = X_centered_z @ Vt_z[:2, :].T

plt.figure(figsize=(12, 8))
plt.scatter(scores_z[:, 0], scores_z[:, 1], alpha=0.5, s=20)

dists_z = np.sqrt(scores_z[:, 0] ** 2 + scores_z[:, 1] ** 2)
threshold_z = np.mean(dists_z) + 2 * np.std(dists_z)
outlier_idx_z = np.where(dists_z > threshold_z)[0]

print("\nOutlier cities (z-score):")
for i in outlier_idx_z:
    plt.annotate(city_names[i], (scores_z[i, 0], scores_z[i, 1]), fontsize=7)
    print(f"  {city_names[i]}: PC1={scores_z[i,0]:.3f}, PC2={scores_z[i,1]:.3f}")

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA Scatter Plot (Z-score normalized)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('p3_zscore_scatter.png', dpi=150, bbox_inches='tight')
plt.show()

# Comparison
print("\n--- Comparison ---")
print("With log-transform, PC1 is dominated by Arts — cities with rich cultural")
print("infrastructure dominate. With z-score, PC1 loads more evenly across all")
print("factors, representing overall city quality. The z-score approach prevents")
print("high-variance factors from dominating and treats all factors equally.")
