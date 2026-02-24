import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import statsmodels.api as sm
from statsmodels.base.model import GenericLikelihoodModel
from scipy.special import betaln
from flot_feature_engineering import *
from flot_prediction import *
from flot_util import *
from flot_visualization import *


# --- Define BetaPrimeReg using GenericLikelihoodModel ---
class BetaPrimeReg(GenericLikelihoodModel):
    def nloglikeobs(self, params):
        # exog includes constant
        exog = self.exog
        endog = self.endog
        k = exog.shape[1]
        # split params into alpha and beta coefficients
        alpha_coefs = params[:k]
        beta_coefs  = params[k:]
        # linear predictors
        eta_a = exog @ alpha_coefs
        eta_b = exog @ beta_coefs
        # shape parameters >0
        a = np.exp(eta_a)
        b = np.exp(eta_b)
        # log-pdf of BetaPrime: a*log(y) - (a+b)*log1p(y) - ln B(a,b)
        ll = a * np.log(endog) - (a + b) * np.log1p(endog) - betaln(a, b)
        return -ll  # negative log-likelihood per observation

    def fit(self, start_params=None, **kwargs):
        if start_params is None:
            # initialize at zeros
            start_params = np.zeros(2 * self.exog.shape[1])
        return super().fit(start_params=start_params, **kwargs)

# --- Load & clean data as before ---
df = pd.read_csv(r'C:\Users\rcpsi\OneDrive\Documents\langchain\Flotation\dados\Flotacao_Dados_Final.csv',sep=';')
#df = pd.read_csv('OneDrive/Documents/langchain/Flotation/dados/Flotacao_Dados_Final.csv', sep=';')
df = parse_time(df)
df = remove_missing(df)
df = filter_by_ph(df, threshold=6)
df_clean = drop_consecutive_duplicates_tolerance(df)
drops = ['inicio','fim','conc_fe','operacao','duration_min']
df_clean.dropna(inplace=True)
df_clean = df_clean.drop(columns=drops).reset_index(drop=True)

# Features & target
features = [col for col in df_clean.columns if col != 'conc_silica']
X = df_clean[features].values
y = df_clean['conc_silica'].values

# Scale and split
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Prepare DataFrames with constant
X_train_df = pd.DataFrame(X_train, columns=features)
X_train_df = sm.add_constant(X_train_df)
X_test_df = pd.DataFrame(X_test, columns=features)
X_test_df = sm.add_constant(X_test_df)

# --- Fit BetaPrime regression ---
model = BetaPrimeReg(y_train, X_train_df)
res = model.fit(method='bfgs', maxiter=1000, disp=False)
print(res.summary())

# --- Predict by computing E[Y] = a/(b-1) ---
params = res.params
k = X_train_df.shape[1]
alpha_coefs = params[:k]
beta_coefs  = params[k:]
# test linear predictors
eta_a_test = X_test_df.values @ alpha_coefs
eta_b_test = X_test_df.values @ beta_coefs
a_test = np.exp(eta_a_test)
b_test = np.exp(eta_b_test)
y_pred = a_test / (b_test - 1)

# Evaluate
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred) * 100
print(f"BetaPrimeReg MAE: {mae:.4f}")
print(f"BetaPrimeReg MAPE: {mape:.2f}%")

# 1) Fit an OLS to log1p(y) to get initial betas
X_const = sm.add_constant(X_train_df)        # DataFrame with constant + features
ols = sm.OLS(np.log1p(y_train), X_const).fit()

# 2) Use those parameters to initialize both alpha and beta
p = X_const.shape[1]
start = np.concatenate([ols.params.values, ols.params.values])  # length 2p

# 3) Re-fit BetaPrimeReg with a good start
bp_model = BetaPrimeReg(y_train, X_const)
res = bp_model.fit(start_params=start, method='bfgs', maxiter=2000, disp=True)

print(res.summary())


import statsmodels.api as sm

glm = sm.GLM(y_train, X_const, 
             family=sm.families.Gamma(link=sm.families.links.log()))
glm_res = glm.fit()
print(glm_res.summary())

y_pred_glm = glm_res.predict(sm.add_constant(X_test_df))
from sklearn.metrics import mean_absolute_error
print("Gamma GLM MAE:", mean_absolute_error(y_test, y_pred_glm))
