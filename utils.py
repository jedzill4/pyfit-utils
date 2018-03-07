from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import lmfit as lmf
import numpy as np


def eval_pol(x,coef,degree):
    result = np.zeros_like(x)
    for i in range(0,degree+1):
        result += coef[i]*np.power(x,i)
    return result

def guess_with_ml(data, x, model, degree):
    sk_model = Pipeline([('poly', PolynomialFeatures(degree=degree)),
              ('linear', LinearRegression(fit_intercept=False))])
    sk_model = sk_model.fit(x[:, np.newaxis], data)
    RR = sk_model.score(x[:,np.newaxis],data)
    coef = sk_model.named_steps['linear'].coef_

    sk_y = eval_pol(x,coef,degree)

    return model.guess(data=sk_y,x=x)
