import pandas as pd
import numpy as np

df = pd.read_csv("Data\\ACSDP5Y2024.DP03-Data.csv")

'Employment & Work-Scheule Friction'

keep = [
    "DP03_0001E",  # Population 16+ (baseline denominator)
    "DP03_0002E",  # In labor force
    "DP03_0003E",  # Civilian labor force
    "DP03_0004E",  # Employed
    "DP03_0005E",  # Unemployed
    "DP03_0007E",  # Not in labor force
]


'Transportation Friction'

keep += [
    "DP03_0019E",  # Drove alone
    "DP03_0020E",  # Carpooled
    "DP03_0021E",  # Public transportation
    "DP03_0022E",  # Walked
    "DP03_0023E",  # Other means
    "DP03_0024E",  # Worked from home
    "DP03_0025E",  # Mean travel time to work (minutes)
]

'Income & Economic Friction'

keep += [
    "DP03_0051E",  # Total households
    "DP03_0052E",  # < $10k
    "DP03_0053E",  # $10k–$14,999
    "DP03_0054E",  # $15k–$24,999
    "DP03_0055E",  # $25k–$34,999
    "DP03_0056E",  # $35k–$49,999
    "DP03_0057E",  # $50k–$74,999
    "DP03_0058E",  # $75k–$99,999
    "DP03_0059E",  # $100k–$149,999
    "DP03_0060E",  # $150k–$199,999
    "DP03_0061E",  # $200k+
    "DP03_0062E",  # Median household income
    "DP03_0063E",  # Mean household income
    "DP03_0064E",  # Households with earnings
    "DP03_0065E",  # Mean earnings
]

'Social Safety net indicators'

keep += [
    "DP03_0066E",  # With Social Security
    "DP03_0067E",  # Mean Social Security income
    "DP03_0068E",  # With retirement income
    "DP03_0069E",  # Mean retirement income
    "DP03_0070E",  # With SSI
    "DP03_0071E",  # Mean SSI income
    "DP03_0072E",  # With cash public assistance
    "DP03_0073E",  # Mean cash public assistance income
    "DP03_0074E",  # With SNAP benefits
]

'Health insurance Friction'

keep += [
    "DP03_0095E",  # Civilian noninstitutionalized population
    "DP03_0096E",  # With health insurance
    "DP03_0097E",  # With private insurance
    "DP03_0098E",  # With public coverage
    "DP03_0099E",  # No insurance
    "DP03_0100E",  # Under 19
    "DP03_0101E",  # Under 19 uninsured
    "DP03_0102E",  # Age 19–64
    "DP03_0103E",  # 19–64 in labor force
    "DP03_0104E",  # 19–64 employed
    "DP03_0105E",  # Employed with insurance
    "DP03_0106E",  # Employed with private insurance
    "DP03_0107E",  # Employed with public coverage
    "DP03_0108E",  # Employed uninsured
    "DP03_0109E",  # 19–64 unemployed
    "DP03_0110E",  # Unemployed with insurance
    "DP03_0111E",  # Unemployed with private insurance
    "DP03_0112E",  # Unemployed with public coverage
    "DP03_0113E",  # Unemployed uninsured
    "DP03_0114E",  # 19–64 not in labor force
    "DP03_0115E",  # Not in labor force with insurance
    "DP03_0116E",  # Not in labor force with private insurance
    "DP03_0117E",  # Not in labor force with public coverage
]


keep = ["GEO_ID", "NAME"] + keep


df_small = df[keep].copy()
numeric_cols = [col for col in df_small.columns if col not in ["GEO_ID", "NAME"]]
df_small[numeric_cols] = df_small[numeric_cols].apply(pd.to_numeric, errors='coerce')

annotation_codes = {
    -666666666: np.nan,   # insufficient sample
    -999999999: np.nan,   # not displayed
    -888888888: np.nan,   # not applicable
}

df_small.replace(annotation_codes, inplace=True)

df_small.fillna(0, inplace=True)

print("Final dataset shape:", df_small.shape)
print(df_small.head())