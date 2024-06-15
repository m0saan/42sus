
import pandas as pd
import matplotlib.pyplot as plt

def separate_houses(data):
    Ravenclaw = data[data["Hogwarts House"] == "Ravenclaw"]
    Gryffindor = data[data["Hogwarts House"] == "Gryffindor"]
    Slytherin = data[data["Hogwarts House"] == "Slytherin"]
    Hufflepuff = data[data["Hogwarts House"] == "Hufflepuff"]
    return Ravenclaw, Gryffindor, Slytherin, Hufflepuff


if __name__ == "__main__":
    input_file = "../datasets/dataset_train.csv"

    data = pd.read_csv(input_file)
    print("len data : ", len(data), "\n")

    Ravenclaw, Gryffindor, Slytherin, Hufflepuff = separate_houses(data)

    Ravenclaw_metrics = Ravenclaw.describe().loc[["mean", "std"]]
    Gryffindor_metrics = Gryffindor.describe().loc[["mean", "std"]]
    Slytherin_metrics = Slytherin.describe().loc[["mean", "std"]]
    Hufflepuff_metrics = Hufflepuff.describe().loc[["mean", "std"]]

    Ravenclaw_metrics = Ravenclaw_metrics.drop('Index', axis=1)
    Gryffindor_metrics = Gryffindor_metrics.drop('Index', axis=1)
    Slytherin_metrics = Slytherin_metrics.drop('Index', axis=1)
    Hufflepuff_metrics = Hufflepuff_metrics.drop('Index', axis=1)

    # sort by standard deviation
    Ravenclaw_metrics = Ravenclaw_metrics.T.sort_values(by="std")
    Gryffindor_metrics = Gryffindor_metrics.T.sort_values(by="std")
    Slytherin_metrics = Slytherin_metrics.T.sort_values(by="std")
    Hufflepuff_metrics = Hufflepuff_metrics.T.sort_values(by="std")

    # Gryffindor_courses = Gryffindor_metrics.index
    # Slytherin_courses = Slytherin_metrics.index
    # Hufflepuff_courses = Hufflepuff_metrics.index

    fig, ax = plt.subplots(figsize=(12, 8))
    bar_width = 0.2
    courses = Ravenclaw_metrics.index

    r1 = range(len(Ravenclaw_metrics['std']))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    r4 = [x + bar_width for x in r3]

    plt.bar(r1, Ravenclaw_metrics['std'], color='b', width=bar_width, edgecolor='grey', label='Ravenclaw')
    plt.bar(r2, Gryffindor_metrics['std'], color='r', width=bar_width, edgecolor='grey', label='Gryffindor')
    plt.bar(r3, Hufflepuff_metrics['std'], color='y', width=bar_width, edgecolor='grey', label='Hufflepuff')
    plt.bar(r4, Slytherin_metrics['std'], color='g', width=bar_width, edgecolor='grey', label='Slytherin')

    plt.xlabel('Courses', fontweight='bold')
    plt.ylabel('Standard Deviation')
    plt.xticks([r + bar_width for r in range(len(Ravenclaw_metrics['std']))], courses, rotation=90)
    plt.title('Standard Deviation of Course Scores by House')
    plt.ylim(0, 100) 
    plt.legend()
    plt.show()