import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap

#classifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

df = pd.read_csv('employee_shortlisting.csv')
df.head(3)

#scatter plot
plt.figure(figsize=(8,6))
colors = {'Yes':'green','No':'red'}

for label in df['Shortlisted'].unique():
    subset = df[df['Shortlisted'] == label]
    plt.scatter(subset['Interview_Score'],subset['Skills_Matched'],c=colors[label],label=label,alpha=0.7,edgecolors='k')

plt.title("Candidate shortlisting Scatter plot")
plt.xlabel("Interview Score")
plt.ylabel("Skills Matched")
plt.legend(title="Shortlisted")
plt.grid(True)
plt.show()


#Feature and Target : (feature engineering)
df['Shortlisted'] = df['Shortlisted'].map({'Yes':1,'No':0})
X = df[["Interview_Score","Skills_Matched"]].values.astype(float)
y = df["Shortlisted"].values.astype(int)

#split data
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#Define Classifier
models = {
    "Logistic Regression":LogisticRegression(),
    "KNN":KNeighborsClassifier(n_neighbors = 3),
    "Decision Tree" :DecisionTreeClassifier(max_depth=5,random_state=42),
    "Random Forest" :RandomForestClassifier(n_estimators=100,max_depth=5,random_state=42),
    "svc":SVC(kernel='rbf')
}

for name,model in models.items():
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    print(name,"Accuracy Score",round(accuracy_score(y_test,y_pred),2))


#creating mesh grid for ploting 
x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
xx,yy=np.meshgrid(np.arange(x_min,x_max,0.01),
                 np.arange(y_min,y_max,0.01))

#plot the Decision Boundries
fig,axes = plt.subplots(2,3,figsize=(18,10))
axes = axes.ravel()
cmap_light = ListedColormap(['#FFAAAA','#AAFFAA'])
cmap_bold = ['red','green']

for idx ,(name,model) in enumerate(models.items()):
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    print(name,"Accuracy Score",round(accuracy_score(y_test,y_pred),2))

    Z = model.predict(np.c_[xx.ravel(),yy.ravel()])
    Z = Z.reshape(xx.shape)

    axes[idx].contourf(xx,yy,Z,alpha=0.3,cmap=cmap_light)
    scatter = axes[idx].scatter(X[:,0],X[:,1],c=y,cmap=ListedColormap(cmap_bold),edgecolor='k',alpha=0.8)
    axes[idx].set_title(name)
    axes[idx].set_xlabel("Interview Score")
    axes[idx].set_ylabel("Skills Matched")

#hide unused subplot
fig.delaxes(axes[-1])
#add legend
fig.legend(*scatter.legend(),loc='lower-center',ncol=2,title="Shortlisted")

plt.tight_layout()
plt.show()


from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv('marketing_spend.csv')
df.head(3)

X=df[['Marketing_Spend_1000s','Employee_Count','Product_Rating']]
y=df['Quarterly_Revenue']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#define models 
models = {
    "KNN" : KNeighborsRegressor(n_neighbors = 5),
    "Decision Tree" : DecisionTreeRegressor(max_depth=3,random_state=42),
    "Random Forest" : RandomForestRegressor(max_depth=5,random_state=42),
    "SVR" : SVR(kernel='linear')
}


for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"{name}:")
    print(f"  R2 Score: {round(r2, 2)}")
    print(f"  MAE: {round(mae, 2)}")
    print(f"  MSE: {round(mse, 2)}\n")