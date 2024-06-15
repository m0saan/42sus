import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

input_file = "../datasets/dataset_train.csv"
data = pd.read_csv(input_file)

data.drop(['Hogwarts House', 'First Name', 'Last Name', 'Birthday','Best Hand'], axis=1, inplace=True)

plt.scatter(data['Arithmancy'], data['Astronomy'])
plt.title('Scatter Plot between Feature 1 and Feature 2')
plt.xlabel('Arithmancy')
plt.ylabel('Astronomy')
plt.show()

features = ['Arithmancy', 'Astronomy', 'Herbology', 'Defense Against the Dark Arts', 'Divination', 'Muggle Studies','Ancient Runes', 'History of Magic', 'Transfiguration', 'Potions', 'Care of Magical Creatures', 'Charms', 'Flying']
data = data[features]

sns.pairplot(data)
plt.suptitle('Pairwise Scatter Plots of Hogwarts Courses', y=1.02)
plt.show()
