import pandas as pd
#Filename
f=pd.read_csv('amazon.graph.large.csv')
import powerlaw
results = powerlaw.Fit((f['count']))
print(results.power_law.alpha)
