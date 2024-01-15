#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from keras.models import Sequential
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

df = pd.read_excel('TSLA LSTM.xlsx')
df = df[['Symbol','Date','Open','High','Low','Close','Adj Close','Volume']]
df['returns'] = df.Close.pct_change()
df['log_returns'] = np.log(1+ df['returns'])


import matplotlib.pyplot as plt

plt.figure(1, figsize = (16,4))
plt.plot(df.log_returns)

grouped_data = df.groupby("Symbol")

# Loop through each group and plot the data
plt.figure(1, figsize=(16, 4))
for symbol, group in grouped_data:
    plt.plot(group["log_returns"], label=symbol)

# Add labels and legend
plt.xlabel("Date")
plt.ylabel("Log Returns")
plt.title("Log Returns for Different Symbols")
plt.legend()

# Show the plot
plt.show()


# ### Preprocessing Steps

df.dropna(inplace = True)
X = df[['Close','log_returns']].values

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range = (0,1)).fit(X)
X_scaled = scaler.transform(X)

X_scaled[:5]

y = [x[0] for x in X_scaled]

y[:5]


split = int(len(X_scaled) * 0.7 )

print(split)

x_train = X_scaled[:split]
x_test = X_scaled[split : len(X_scaled)]

y_train = y[:split]
y_test = y[split : len(y)]

assert len(x_train) == len(y_train)
assert len(x_test) == len(y_test)


n =3 

xtrain = []
ytrain = []

xtest = []
ytest = []

for i in range(n, len(x_train)):
    xtrain.append(x_train[i - n : i, : x_train.shape[1]])
    ytrain.append(y_train[i])  # predict next record
    
for i in range(n, len(x_test)):
    xtest.append(x_test[i - n : i, : x_test.shape[1]])
    ytest.append(y_test[i])  # predict next record
    
ytest

xtrain[0]
val = np.array(ytrain[0])
val = np.c_[val, np.zeros(val.shape)]

scaler.inverse_transform(val)


xtrain, ytrain = (np.array(xtrain), np.array(ytrain))
xtrain  = np.reshape( xtrain, (xtrain.shape[0], xtrain.shape[1],xtrain.shape[2]))
                  
xtest, ytest = (np.array(xtest), np.array(ytest))
xtest  = np.reshape( xtest, (xtest.shape[0], xtest.shape[1], xtest.shape[2]))


print(xtrain.shape)
print(ytrain.shape)

print("----")

print(xtest.shape)
print(ytest.shape)


from keras.models import Sequential
from keras.layers import LSTM, Dense

model = Sequential()

model.add(LSTM(4, input_shape = (xtrain.shape[1], xtrain.shape[2])))

model.add(Dense(1))

model.compile(loss = "mean_squared_error",optimizer = "adam")

model.fit(
xtrain, ytrain, epochs = 25, validation_data = (xtest, ytest), batch_size = 16, verbose = 1
)

model.summary()


trainpredict = model.predict(xtrain)

testpredict = model.predict(xtest)

trainpredict = np.c_[trainpredict,np.zeros(trainpredict.shape)]

testpredict = np.c_[testpredict,np.zeros(testpredict.shape)]

## Invert Predictions

trainpredict = scaler.inverse_transform(trainpredict)
trainpredict = [x[0] for x in trainpredict]

testpredict = scaler.inverse_transform(testpredict)
testpredict = [x[0] for x in testpredict]


print(trainpredict[:5])
print(testpredict[:5])


## calculate mean squared error 

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt

trainscore_mse = mean_squared_error([x[0][0]  for x in xtrain], trainpredict, squared = False )
trainscore_rmse = sqrt(trainscore_mse)
trainscore_mae = mean_absolute_error([x[0][0]  for x in xtrain], trainpredict)
r2_train = r2_score([x[0][0]  for x in xtrain], trainpredict)

print("Train Score: %.2f MSE" % (trainscore_mse))
print("Train Score: %.2f RMSE" % (trainscore_rmse))
print("Train Score: %.2f MAE" % (trainscore_mae))
print("Train Score: %.2f R2 Score" % (r2_train))

testscore_mse = mean_squared_error([x[0][0]  for x in xtest], testpredict, squared = False )
testscore_rmse = sqrt(testscore_mse)
testscore_mae = mean_absolute_error([x[0][0]  for x in xtest], testpredict)
r2_test = r2_score([x[0][0]  for x in xtest], testpredict)

print("Test Score: %.2f RMSE" % (testscore_mse))
print("Test Score: %.2f RMSE" % (testscore_rmse))
print("Test Score: %.2f MAE" % (testscore_mae))
print("Test Score: %.2f R2 Score" % (r2_test))



import matplotlib.pyplot as plt

import matplotlib.pyplot as plt


# Set figure size and create axes object
fig, ax = plt.subplots(figsize=(16, 8))

# Plot actual and predicted values
ax.plot(df['Date'], df['Close'], label='Actual')
ax.plot(df.iloc[split+n:]['Date'], testpredict, label='Predicted')

# Add labels and legend
ax.set_title('Actual vs. Predicted Values for TSLA')
ax.set_xlabel('Date')
ax.set_ylabel(' Close Price ($)')
ax.legend()

# Format x-axis date labels
ax.xaxis.set_major_locator(plt.MaxNLocator(8))

# Show the plot
plt.show()


import pandas as pd

# Load original data into a DataFrame
df = pd.read_excel('WMT LSTM.xlsx', index_col='Date', parse_dates=True)

# Extract the dates from the test data
dates = df.index[-len(testpredict):]

# Create a new DataFrame with the predicted values
predictions = pd.DataFrame({'Predictions': testpredict}, index=dates)

# Concatenate the original "Close" column with the predicted values
result = pd.concat([df['Close'], predictions], axis=1)

result = result.tail(30)

result.to_csv('TSLA close and predicted prices Updated.csv')
