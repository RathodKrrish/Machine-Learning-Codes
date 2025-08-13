#Problem 1:
'''find  that if value is in x is toxic or not ,uppon y'''

import numpy as np
from sklearn import linear_model

x= np.array([1.2,0.5,0.8,1.0,1.4,0.6,3.0,2.8,3.2,3.5,2.9,3.1]).reshape(-1,1)
y= np.array([0,0,0,0,1,1,1,1,1,1,1,1])

logr = linear_model.LogisticRegression()
logr.fit(x,y)

test_value = np.array([1.2]).reshape(-1,1)
predicted = logr.predict(test_value)
probability = logr.predict_proba(test_value)

print(f"Predict class (0 = Non-Toxic & 1 = Toxic) : {predicted[0]}")
print(f"Probability of toxicity is  : {probability[0][1]}")





#problem 2 :
'''Find that if Customers sees a product X times,will they Buy or not'''

import numpy as np
from sklearn import linear_model

X = np.array([1,2,3,4,5,6,7,8,9,10]).reshape(-1,1)
Y = np.array([0,0,0,0,1,1,1,1,1,1])

model = linear_model.LogisticRegression()
model.fit(X,Y)

test_value = np.array([6]).reshape(-1,1)
predicted = model.predict(test_value)
probability = model.predict_proba(test_value)

print(f"Customer Purchases Prediction : {predicted[0]}")
print(f"Probability of Purchases  : {probability[0][1]}")