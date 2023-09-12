# Mengimpor library pendukung
import pandas as pd

# Menyimpan file CSV dalam DataFrame
df = pd.read_csv('../data/bitcoin_monthly_historical_data.csv')

# Menghapus atribut Adj Close dan Volume
df.drop(labels='Adj Close', axis=1, inplace=True)
df.drop(labels='Volume', axis=1, inplace=True)

# Mengubah format atribut Date menjadi YYYYMMDD
df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y%m%d')

# Membulatkan nilai atribut Open, High, Low, dan Close
df['Open'] = df['Open'].round(decimals=2)
df['High'] = df['High'].round(decimals=2)
df['Low'] = df['Low'].round(decimals=2)
df['Close'] = df['Close'].round(decimals=2)

# Menyimpan DataFrame dalam file CSV baru
df.to_csv('../dataset/monthly_dataset.csv', index=False)
