import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd

df = pd.read_csv("thresholding.csv")

pixels = pd.to_numeric(df['Pixels remaining'], errors='coerce').dropna()

# plt.figure(figsize=(10, 4))
# plt.hist(pixels, bins=50, color='skyblue', edgecolor='black')
# plt.title('Raw Pixels Remaining Distribution')
# plt.xlabel('Pixels Remaining')
# plt.ylabel('Frequency')
# plt.grid(True)
# plt.tight_layout()
# plt.show()

log_pixels = np.log1p(pixels)
plt.figure(figsize=(10, 4))
plt.hist(log_pixels, bins=50, color='skyblue', edgecolor='black')
plt.title('Log-Transformed Pixels Remaining Distribution')
plt.xlabel('log(1 + Pixels Remaining)')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.show()

print(log_pixels.dtype)
