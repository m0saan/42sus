import pandas as pd

def calcul_count(data):
    count = 0
    for i in range(len(data)):
        if not pd.isnull(data[i]):
            count += 1
    return format(count, '.6f')

def calcul_mean(data):
    count = 0
    sum = 0
    for i in range(len(data)):
        if not pd.isnull(data[i]):
            count += 1
            sum += data[i]
    mean = sum / count if count != 0 else 0
    return format(mean, '.6f')

def calcul_std(data):
    mean = float(calcul_mean(data))
    count = 0
    sum_square_deviation = 0

    for i in range(len(data)):
        if not pd.isnull(data[i]):
            sum_square_deviation += (data[i] - mean) ** 2
            count += 1
    variance = sum_square_deviation / (count - 1) if count > 1 else 1

    return format(variance ** 0.5, ".6f") 

def calcul_min(data):
    min = data[0]
    for i in range(1, len(data)):
        if not pd.isnull(data[i]) and data[i] < min:
            min = data[i]
    return format(min, '.6f')

import numpy as np

def calculate_percentile(data, percentile):
    data_sorted = data.dropna().sort_values()

    k = (len(data_sorted) - 1) * percentile / 100
    f = int(np.floor(k))
    c = k - f

    if f == len(data_sorted) - 1:
        res = data_sorted.iloc[-1]
    else:
        res = data_sorted.iloc[f] * (1 - c) + data_sorted.iloc[f + 1] * c

    return format(res, '.6f')

def calcul_max(data):
    max = data[0]
    for i in range(1, len(data)):
        if not pd.isnull(data[0]) and data[i] > max:
            max = data[i]
    return format(max, '.6f')

if __name__ == "__main__":
    input_file = "../datasets/dataset_train.csv"
    data = pd.read_csv(input_file)
    num_data = data[["Arithmancy","Astronomy","Herbology","Defense Against the Dark Arts","Divination","Muggle Studies","Ancient Runes","History of Magic","Transfiguration","Potions","Care of Magical Creatures","Charms","Flying"]]

    G_mean = []
    G_count = []
    G_std = []
    G_min = []
    G_25 = []
    G_50 = []
    G_75 = []
    G_max = []

    # Compute for each feature
    for col in num_data.columns:
        G_mean.append(calcul_mean(num_data[col]))
        G_count.append(calcul_count(num_data[col]))
        G_std.append(calcul_std(num_data[col]))
        G_min.append(calcul_min(num_data[col]))
        G_25.append(calculate_percentile(num_data[col], 25))
        G_50.append(calculate_percentile(num_data[col], 50))
        G_75.append(calculate_percentile(num_data[col], 75))
        G_max.append(calcul_max(num_data[col]))

    # Create a DataFrame to store the results
    stats = pd.DataFrame({
        "Feature": num_data.columns,
        "Count": G_count,
        "Mean": G_mean,
        "Std": G_std,
        "Min": G_min,
        "25%": G_25,
        "50%": G_50,
        "75%": G_75,
        "Max": G_max
    })
    
    stats = stats.set_index("Feature")
    print(stats.transpose())