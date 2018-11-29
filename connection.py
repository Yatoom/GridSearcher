from pymongo import MongoClient


def connect():
    client = MongoClient('mongodb://...:...@ds245687.mlab.com:45687/experiment-60')
    database = client.get_database("experiment-60")
    return client, database["experiments"]

