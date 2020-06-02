import pandas as pd

path = "/home/tushar/Desktop/runpro/SoC-IA/Assgn-5/Songs_LA - Sheet1.csv"
df = pd.read_csv(path)

df["Popularity(youtube views-in millions)"] /= 5000

print(df["Popularity(youtube views-in millions)"].max())

df.to_csv(path)
