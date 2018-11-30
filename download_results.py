import connection
import pandas as pd

client, db = connection.connect()

frame = pd.DataFrame(list(db.find()))
frame.to_csv("result.csv")