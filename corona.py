import numpy as np
from statsmodels.api import OLS
from statsmodels.tools.tools import add_constant
from matplotlib import pyplot as plt
confirmed = np.array([
    45,
    62,
    121,
    198,
    291,
    440,
    571,
    830,
    1287,
    1975,
    2744,
    4515,
    5974,
    7711,
    9692,
    11791,
    14380,
    17205,
    20438,
    24324,
    28018,
    31161,
    34546,
    37198,
])
x = np.arange(len(confirmed))
x = add_constant(x)
model = OLS(np.log(confirmed[:14]), x[:14]) 
result = model.fit()
result.summary()
plt.plot(
    np.exp(result.predict(x[:14])),
   
    
    label="Prédiction du fonction exp",
    
       
)
plt.plot(confirmed[:14], ".", label="Cas réels, CN")
plt.legend()
plt.xlabel("jours")
plt.ylabel("nombres de malades")
plt.show()
world_population = 7763252653
days = 0
infected = confirmed[14]
while infected < world_population:
    days += 1
    infected = np.exp(result.predict([1, 13 + days]))[0]
print(f"Number of days until whole world is infected: {days}")
plt.plot(np.exp(result.predict(x[:16])))
plt.plot(confirmed[:16], ".")
plt.show()
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
logistic_function = lambda x, a, b, c, d: \
    a / (1 + np.exp(-c * (x - d))) + b
confirmed = np.array(confirmed)
x = x[:, 1]
(a_, b_, c_, d_), _ = curve_fit(logistic_function, x, confirmed)
confirmed_pred = logistic_function(x, a_, b_, c_, d_)
plt.plot(x, confirmed_pred, label="Prédiction du fonction logistique")
plt.legend()
plt.title(" Réalisé Par Jalal",color="red")
plt.show()

