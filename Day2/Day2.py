import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
pd.options.display.max_columns = 7

# reading the csv file
df = pd.read_csv('../titanic.csv')

# prints first 5 row
print(df.head())

# prints dimesion
print(df.shape)

# detailed info of each feature
print(df.describe())

# pandas series
col = df['Pclass']
print(col)

# selecting multiple feature
col2 = df[['Pclass','Age','Fare']]
print(col2)

# creating new feeature
df['Male'] = df['Sex'] == 'male'
print(df)

# convert to numpy array
print(df['Age'].values)


arr = df[['Pclass','Age','Fare']].values

print(arr.shape)

# first row, second column
print(arr[0,1])

# 3rd row
print(arr[2])

# 3rd column only
print(arr[:,2])

# creating a mask
mask = arr[:,1] < 18

print(mask)

# using the mask
print(arr[mask])

# count
print(mask.sum())


# plt.scatter(df['Age'],df['Fare'],c=df['Pclass'])
# plt.xlabel('Age')
# plt.ylabel('fare')
# plt.plot([0,80],[85,20])

list = [4, 6, 1, -3, 8]
print(np.mean(list)) # mean
# list.sort()
# print(list)
print(np.median(list)) # median
# 50th percentile
print(np.percentile(list,50))
# 25th percentile
print(np.percentile(list,25))
# 75th percentile
print(np.percentile(list,75))
# variance
print(np.var(list))
# standard deviation
print(np.std(list))
