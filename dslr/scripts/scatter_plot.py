import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

input_file = "../datasets/dataset_train.csv"
data = pd.read_csv(input_file)

features = ['Astronomy', 'Defense Against the Dark Arts']
data = data[features]

sns.pairplot(data)
plt.suptitle('Pairwise Scatter Plots of Hogwarts Courses', y=1.02)  # Adjust the title position
plt.show()