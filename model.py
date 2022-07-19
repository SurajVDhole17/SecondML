from pandas import *
from matplotlib.pyplot import *
from seaborn import *
from sklearn.linear_model import LinearRegression
import pickle
import joblib
df = read_csv("dataset.csv")
df.head()
jointplot(x="YearsExperience", y="Salary", data = df, kind = "reg")
show()
x = df[["YearsExperience"]]
y = df[["Salary"]]
lm = LinearRegression()
lm.fit(x,y)
lm.predict([[9]])
joblib.dump(lm,"modelpred.pkl")
model = joblib.load("modelpred.pkl")
print(model.predict([[5]]))