import pandas as pd
import numpy as np
import gspread
from oauth2client.service_account import ServiceAccountCredentials

scope = ["https://spreadsheets.google.com/feeds", 'https://www.googleapis.com/auth/drive']

creds = ServiceAccountCredentials.from_json_keyfile_name("creds.json", scope)

client = gspread.authorize(creds)

sheet = client.open("testing").sheet1
#sheet.delete_col(1)

df = pd.DataFrame(sheet.get_all_records())


headers = df.columns.tolist()
if not headers[0]:
    df = df[headers[1:]]
    headers = headers[1:]


index = list(range(1, 62))
index.reverse()
for i in index:
    print(i)
    sheet.delete_row(i)


for j in range(len(headers)):
    headers[j] = str(headers[j])
sheet.append_row(headers)

for i in range(60):
    new_row = df.iloc[i].values.tolist()
    for p in range(len(new_row)):
        if p == 0:
            new_row[0] = str(new_row[0])
        elif p == 4:
            new_row[4] = float(new_row[4])
        else:
            new_row[p] = int(new_row[p])
    sheet.append_row(new_row)
