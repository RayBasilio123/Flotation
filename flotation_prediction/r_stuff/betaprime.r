# install required packages (only once)
if (!requireNamespace("devtools", quietly=TRUE)) install.packages("devtools")
if (!requireNamespace("gamlss",   quietly=TRUE)) install.packages("gamlss")
# BPmodel contains the BP family for gamlss
devtools::install_github("santosneto/BPmodel")

library(gamlss)
library(BPmodel)    # registers the BP() family
library(dplyr)      # for data‐wrangling
library(Metrics)    # for MAE, MAPE

# 1) Load & basic cleaning
df <- read.csv("Flotacao_Dados_Final.csv", sep=";")

# parse timestamps & compute duration
df <- df %>% 
  mutate(
    inicio       = as.POSIXct(inicio, "%Y-%m-%d %H:%M:%OS"),
    fim          = as.POSIXct(fim,    "%Y-%m-%d %H:%M:%OS"),
    duration_min = as.numeric(difftime(fim, inicio, units="mins"))
  )

# coerce operationals to numeric, drop any remaining NA
ops <- setdiff(names(df), c("inicio","fim","operacao"))
df[ops] <- lapply(df[ops], function(x) as.numeric(as.character(x)))
df <- na.omit(df)

# filter pH ≥ 6
df <- df %>% filter(
  ph_flotacao_linha01 >= 6,
  ph_flotacao_linha02 >= 6
)

# drop “stuck” repeats in conc_silica (tolerance 0.01)
df <- df %>% arrange(inicio) %>% 
  filter(
    row_number() == 1 |
    abs(conc_silica - lag(conc_silica)) >= 0.01
  )

# 2) Specify your features
# replace these with whichever operational columns you want
base_features <- c(
  "param_dosagem_amido",
  "dosagem_amina_conc_magnetica",
  "ph_flotacao_linha01",
#  "ph_flotacao_linha02",
  "densidade_alimentacao_flotacao",
  "nivel_celula_li640101",
  "TO_LI6401_02",
#  "nivel_celula_li640201",
#  "nivel_celula_li640202",
#  "nivel_celula_li641101",
#  "nivel_celula_li641102",
#  "nivel_celula_li641201",
#  "nivel_celula_li641202",
#  "nivel_celula_li642101",
#  "nivel_celula_li642201",
#  "nivel_celula_li643101",
#  "nivel_celula_li643201",
  "vazao_alimentacao_flotacao"
)

# discard columns not used
df <- df %>% select(all_of(c(base_features, "conc_silica")))

# 3) Fit mean–precision Beta‐prime regression
#   μ-link = log, φ-link = log (so μ>0, φ>0)
mu.form    <- as.formula(paste("conc_silica ~", paste(base_features, collapse = " + ")))
sigma.form <- ~ 1   # constant precision; or use e.g. "~ ph_flotacao_linha01" to model φ
#sigma.form <- ~ dosagem_amina_conc_magnetica + param_dosagem_amido + vazao_alimentacao_flotacao   # constant precision; or use e.g. "~ ph_flotacao_linha01" to model φ



fit_bp <- gamlss(
  formula      = mu.form,
  sigma.formula= sigma.form,
  family       = BP(mu.link="log", sigma.link="log"),
  data         = df
)

# 4) Summarize
summary(fit_bp)

# 5) Predictions & metrics
y_true <- df$conc_silica
y_pred <- fitted(fit_bp, what="mu")  # E[Y|X] = μ̂

cat("MAE: ", mae(y_true, y_pred), "\n")
cat("MAPE:", mape(y_true, y_pred)*100, "%\n")

# 6) Residual diagnostics
# 6a) Quantile residuals (should be ~ N(0,1))
rq <- resid(fit_bp, type="simple")
plot(rq, main="Quantile Residuals"); abline(h=0, col="red")

# 6b) Pearson residuals
rp <- resid(fit_bp, type="simple")
plot(rp, main="Pearson Residuals"); abline(h=0, col="red")

# 6c) Residuals vs Actual values
residuals <- y_true - y_pred
plot(
  y_true, residuals,
  main   = "Residuals vs Actual conc_silica",
  xlab   = "Actual conc_silica",
  ylab   = "Residual = Actual − Predicted",
  pch    = 16,
  col    = "#2C7BB6"
)
abline(h = 0, lty = 2, col = "red")
