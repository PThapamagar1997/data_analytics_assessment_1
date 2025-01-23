################ Data cleaning the Iris dataset #################
from sklearn import datasets
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# load iris dataset
iris = datasets.load_iris()
# Since this is a bunch, create a dataframe
iris_df=pd.DataFrame(iris.data)
iris_df['class']=iris.target

iris_df.columns=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
#### ===> TASK 1: here - add two more lines of the code to find the number and mean of missing data
print("Number of missing values:", iris_df.isnull().sum().sum())
print("Mean of missing data:", iris_df.isnull().mean().mean())
cleaned_data = iris_df.dropna(how="all", inplace=True) # remove any empty lines


iris_X=iris_df.iloc[:5,[0,1,2,3]]
print(iris_X)

### TASK2: Here - Write a short readme to explain above code and how we can calculate the corrolation amoung featuers with description
# Calculate correlation matrix
correlation_matrix = iris_df[['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid']].corr()

# Print correlation matrix
print("\nCorrelation Matrix:")
print(correlation_matrix)

# Visualize correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()

# Additional analysis: Pairplot to visualize relationships between features
sns.pairplot(iris_df, hue='class')
plt.suptitle('Pairplot of Iris Dataset Features', y=1.02)
plt.show()