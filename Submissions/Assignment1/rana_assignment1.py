import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv('titanic.csv')

twoDimentionalArray = df[['Survived', 'Pclass', 'Sex', 'Age', 'Siblings/Spouses','Parents/Children','Fare']]
print(twoDimentionalArray.values)

plt.scatter(df['Age'], df['Fare'])
plt.show()