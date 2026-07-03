import pandas as pd


df  = pd.read_csv('data\sample_10000_rows.csv',nrows = 1000,usecols =
[
    'passenger_count',
    'trip_distance',
    'fare_amount',
    'tip_amount',
    'total_amount'
])

df = df.dropna()

df = df[df['fare_amount'] > 0]
df = df[df['trip_distance']>0]
df = df[df['total_amount'] > 0]

df.to_csv('data/taxi_clean.csv', index = False)
print('Data saved')


