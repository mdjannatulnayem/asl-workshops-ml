import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('titanic.csv')
df = pd.DataFrame(data)


print(df[['Survived','Pclass','Sex','Age','Siblings/Spouses','Parents/Children','Fare']].values)
arr = df[['Survived','Pclass','Sex','Age','Siblings/Spouses','Parents/Children','Fare']].values
print(arr.shape)

x = df["Age"]
y = df["Fare"]

plt.scatter(x,y,color="blue")

plt.title("Titanic")
plt.xlabel("X axis")
plt.ylabel("Y axis")

plt.show()
