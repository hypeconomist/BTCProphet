import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fbprophet import Prophet
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from fbprophet.plot import add_changepoints_to_plot

coin = 'bitcoin'

startDate = 20160201
endDate = 20200430

tables = pd.read_html('https://coinmarketcap.com/currencies/'+coin+'/historical-data/?start='+str(startDate)+'&end='+str(endDate))
#print(tables[0])

#reverse dataframe
table = tables[2].iloc[::-1]
#clean dataframe
df = table[['Date', 'Close**']].copy()
#normalize data
df = df.rename(index=str, columns={"Date":"ds", "Close**":"y"})
print(df)

#model
m = Prophet(seasonality_mode='multiplicative')
#m.add_seasonality('self_define_cycle',period=8,fourier_order=8,mode='additive')
m.fit(df)

future = m.make_future_dataframe(periods=365, freq='D')
fcst = m.predict(future)
fig = m.plot(fcst, xlabel = 'Date', ylabel = 'Closing value', uncertainty=True)
a = add_changepoints_to_plot(fig.gca(), m, fcst)

plt.title('Bitcoin forecast')
plt.show()
