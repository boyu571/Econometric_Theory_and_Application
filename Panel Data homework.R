install.packages('clubSandwich')

library(readxl)
library(dplyr)
library(lmtest)
library(sandwich)
library(clubSandwich)
library(plm)

# 1
wage = read.csv("wagedata.csv")
wage
nrow(wage)

# 2
ols1 = plm(lwage ~ educ + black + hisp + exper + expersq + married + union, data = wage, model = 'pooling')
summary(ols1)

coeftest(ols1, vcov=vcovHC)

# 3
wage_re = plm(lwage ~ educ + black + hisp + exper + expersq + married + union, data = wage, model = 'random')
summary(wage_re)
coeftest(wage_re, vcov=vcovHC)

# 4
wage_fe = plm(lwage ~ educ + black + hisp + exper + expersq + married + union, data = wage, model = 'within')
summary(wage_fe)
coeftest(wage_fe, vcov=vcovHC)

# 5
phtest(wage_fe, wage_re)

# 8
wage_ht <- plm(lwage ~ educ + black + hisp + exper + expersq + married + union|
                 black + hisp+exper+expersq+married+union, 
                data = wage, 
                model = "random",
               random.method = "ht",
               inst.method = "baltagi")
summary(wage_ht)

# 9
phtest(wage_fe, wage_ht)

