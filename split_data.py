import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os

file_path = "Downloads/train.csv"
df = pd.read_csv(file_path)
df

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

directory = os.path.dirname(file_path)

train_df.to_csv(os.path.join(directory, 'train_data.csv'), index=False)
test_df.to_csv(os.path.join(directory, 'test_data.csv'), index=False)

sorted_df = df.sort_values(by='Name')

sorted_df.head(50)
