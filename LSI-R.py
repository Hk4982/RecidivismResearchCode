
SAV_PATH = "39307-0001-Data.sav"  

import numpy as np
import pandas as pd
import pyreadstat
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import ttest_ind




df, meta = pyreadstat.read_sav(SAV_PATH)


df['risk_score'] = 0



df.loc[df['AGE'] > 0, 'risk_score'] += 1
df.loc[df['AGE'] > 0, 'risk_score'] += 1
df.loc[df['INCPRE'] == 1, 'risk_score'] += 1
df.loc[df['CHLDHDPOLICETRBL12M'] == 1, 'risk_score'] += 1
df.loc[df['AGE'] > 0, 'risk_score'] += 1
df.loc[df['EMP12M'] == 0, 'risk_score'] += 1
df.loc[(df['EMP12M'] == 0) & (df['EMP6M'] == 0), 'risk_score'] += 1
df.loc[df['JOBSHLD12M'] == 0, 'risk_score'] += 1
df.loc[df['EMPWHYNT2M'] == 14, 'risk_score'] += 1
df.loc[df['EDUCATIONPRE'] <= 3, 'risk_score'] += 1
df.loc[df['EDUCATIONPRE'] <= 4, 'risk_score'] += 1
df.loc[df['CHLDHDSCHLSUSPEXP12M'] == 1, 'risk_score'] += 1
df.loc[df['EMP12M'] == 0, 'risk_score'] += 3
df.loc[(df['EMP12M'] == 1) & (df['SATEMP12M'] > 2), 'risk_score'] += 1
df.loc[(df['EMP12M'] == 1) & (df['SATEMP12M'] > 3), 'risk_score'] += 1 
df.loc[df['AGE'] > 0, 'risk_score'] += 3
df.loc[df['EMP12M'] == 0, 'risk_score'] -= 3
df.loc[(df['EMP12M'] == 1) & (df['NUMPEERS12M'] >= 3), 'risk_score'] -= 1
df.loc[(df['EMP12M'] == 1) & (df['SATFRNDSHIPS6M'] <= 2), 'risk_score'] -= 1
df.loc[(df['EMP12M'] == 1) & (df['SATEMP6M'] <= 2), 'risk_score'] -= 1
df.loc[df['AGE'] > 0, 'risk_score'] += 3
df.loc[df['EMP12M'] == 0, 'risk_score'] -= 3
df.loc[(df['EMP12M'] == 1) & (df['SATEMP12M'] < 3), 'risk_score'] -= 1
df.loc[(df['EMP12M'] == 1) & (df['ACCPTLGLDCSN12M'] <= 2), 'risk_score'] -= 1
df.loc[df['MNYDIFFCLTY12M'] > 3, 'risk_score'] += 1
df.loc[df['MNYDIFFCLTY12M'] > 2, 'risk_score'] += 1
df.loc[df['MNYDIFFCLTY12M'] > 1, 'risk_score'] += 1
df.loc[df['FINSUPPLBCASST12M'] == 1, 'risk_score'] += 1
df.loc[(df['SATFAMRELTN6M'] < 3) & (df['MARITALSTATUS6M'] != 4) & (df['MARITALSTATUS6M'] != 3), 'risk_score'] += 1
df.loc[df['SATFAMRELTN12M'] > 1, 'risk_score'] += 1
df.loc[df['SATFAMRELTN12M'] > 2, 'risk_score'] += 1
df.loc[df['SATFAMRELTN12M'] > 3, 'risk_score'] += 1
df.loc[df['SATFAMRELTN12M'] > 1, 'risk_score'] += 1
df.loc[df['SATFAMRELTN12M'] > 2, 'risk_score'] += 1
df.loc[df['SATFAMRELTN12M'] > 3, 'risk_score'] += 1
df.loc[df['FAMCNVTB'] == 1, 'risk_score'] += 2
df.loc[df['SATHOUSN12M'] > 1, 'risk_score'] += 1
df.loc[df['SATHOUSN12M'] > 2, 'risk_score'] += 1
df.loc[df['SATHOUSN12M'] > 3, 'risk_score'] += 1
df.loc[df['NUMRES12M'] > 2, 'risk_score'] += 1
df.loc[df['NGHPRSNSAFEB'] < 3, 'risk_score'] += 2
df.loc[df['FREQCNTCTFAMFRND12M'] < 5, 'risk_score'] += 2
df.loc[df['NUMPROGRAMS12M'] <= 7, 'risk_score'] += 1
df.loc[df['NUMPEERS12M'] == 0, 'risk_score'] += 2
df.loc[df['FRNDPRSN12M'] >= 1, 'risk_score'] -= 1
df.loc[df['FRNDPRSN12M'] < 3, 'risk_score'] += 2
df.loc[df['FRNDPRSN12M'] < 3, 'risk_score'] += 1
df.loc[df['DRGALCHIST'] == 1, 'risk_score'] += 1
df.loc[(df['ACCSHLTH4DRGTXB'] == 1) | (df['ACCSHLTH4DRGTXPRE'] == 1) | (df['ACCSHLTH4DRGTX12M'] == 1), 'risk_score'] += 1
df.loc[df['ALCOHOL12M'] == 1, 'risk_score'] += 1
df.loc[df['MARIJUANAFREQ12M'] == 3, 'risk_score'] += 1
df.loc[df['MARIJUANAFREQ12M'] == 4, 'risk_score'] += 1
df.loc[df['MARIJUANAFREQ12M'] == 5, 'risk_score'] += 1
df.loc[df['DRGDIFFCLTY12M'] > 2, 'risk_score'] += 1
df.loc[df['ACCSHLTH4DRGTX12M'] == 1, 'risk_score'] += 2        
df.loc[(df['SATHLTH12M'] >= 3) &  (df['ACCSHLTH3MNTL12M'] == 1), 'risk_score'] += 2
df.loc[df['ACCSHLT3HMNTLPRE'] == 1, 'risk_score'] += 1
df.loc[(df['ACCSHLT3HMNTLPRE'] == 1) &  (df['ACCSHLTH3MNTL12M'] == 1), 'risk_score'] += 2
df.loc[df['ACCPTLGLDCSN12M'] < 3, 'risk_score'] += 1
df.loc[df['OBEYLAW12M'] < 3, 'risk_score'] += 1
df.loc[df['CRIMACTIVDIFFCLTY12M'] <= 3, 'risk_score'] += 2
df.loc[df['FOLLOWRULES12M'] > 3, 'risk_score'] += 2
df.loc[df['POLICECRTSENMY12M'] < 3, 'risk_score'] += 1
df.loc[df['POLICETRTFAIR12M'] > 3, 'risk_score'] += 1
df.loc[df['POLICETRTFAIR12M'] < 3, 'risk_score'] += 2
df.loc[df['SUPERVHELP12M'] == 0, 'risk_score'] += 2

#####################
#outcome
df['reincarcerated_1y'] = (df['SUPERVCHNGE12M'] == 4).astype(int)

#race map
race_map = {2: 'Black', 4: 'White', 5: 'Multiracial', 6: 'Other'}
df['RaceLabel'] = df['RACE'].map(race_map)
df = df[df['RaceLabel'].isin(['White','Black'])].copy()
df['race_black'] = (df['RaceLabel'] == 'Black').astype(int)

controls = []

if 'AGE' in df.columns:
    controls.append('AGE')

if 'SEX' in df.columns:
    df['gender_male'] = df['SEX'].map({1: 1, 2: 2})
    if df['gender_male'].notna().any():
        controls.append('gender_male')


# MODEL DATA
needed = ['reincarcerated_1y', 'risk_score', 'race_black'] + controls
model_df = df[needed].dropna().copy()
model_df['risk_score_z'] = (model_df['risk_score'] - model_df['risk_score'].mean()) / model_df['risk_score'].std(ddof=0)


rhs = ['risk_score_z', 'race_black'] + controls
rhs = list(dict.fromkeys(rhs))
formula = 'reincarcerated_1y ~ ' + ' + '.join(rhs)

model = smf.logit(formula=formula, data=model_df).fit(disp=0)



params = model.params
or_table = pd.DataFrame({
    'odds_ratio': np.exp(params),
    'p_value': model.pvalues
})
print("\nOdds Ratios main model per 1 SD risk score:")
print(or_table)

#score × race 
formula_int = formula.replace('risk_score_z', 'risk_score_z * race_black', 1)
model_int = smf.logit(formula=formula_int, data=model_df).fit(disp=0)


params_i = model_int.params
or_table_i = pd.DataFrame({
    'odds_ratio': np.exp(params_i),
    'p_value': model_int.pvalues
})
print("\nOdds Ratios interaction model:")
print(or_table_i)


tab = model_df.groupby('race_black')['reincarcerated_1y'].agg(
    rate='mean', n='count'
)
tab.index = tab.index.map({0:'White', 1:'Black'})
print("\nReincarceration  by race:")
print(tab.round(3))


w = model_df.loc[model_df['race_black']==0, 'risk_score'].dropna()
b = model_df.loc[model_df['race_black']==1, 'risk_score'].dropna()
t_stat, p_score = ttest_ind(w, b, equal_var=False, nan_policy='omit')
print(f"Avg risk score= White: {w.mean():.2f}, Black: {b.mean():.2f}")
sd_oras = model_df['risk_score'].std()
print("Standard deviation:", sd_oras, "LSI points")



