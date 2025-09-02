from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import matplotlib.pyplot as plt

#load data set
iris = load_iris()

#convert dataset to pandas dataframe
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data["species"] = iris.target

print("First 5 rows of dataset:")
print(data.head)

#split dataset to training and testing
x = iris.data
y = iris.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#train model
model = DecisionTreeClassifier()
model.fit(x_train,y_train)

#test model
accuracy = model.score(x_test,y_test)
print("\nModel Accuracy:", accuracy)

#make prediction
sample = [[5.1,3.5,1.4,0.2]]
predicition = model.predict(sample)
print("\nPrediction for sample {}:{}".format(sample, iris.target_names[predicition][0]))

#plot the dataset
plt.scatter(iris.data[:, 0], iris.data[:, 1], c=iris.target, cmap='viridis')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('Iris Dataset Visualization')
plt.show()
