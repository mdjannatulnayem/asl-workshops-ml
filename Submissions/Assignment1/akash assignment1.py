df = pd.read_csv("C:\\Users\\a\\PycharmProjects\\pythonProject\\titanic.csv")
# df.head()
twoDArray = df[['Survived', 'Pclass', 'Sex', 'Age','Fare']]
print(twoDArray.values)
plt.scatter(df['Age'],df['Fare'])
plt.xlabel("Age")
plt.ylabel("Fare")
plt.show()
