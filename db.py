from flask_pymongo import pymongo
from app import app

try:
    CONNECTION_STRING = "mongodb+srv://csnehachandrasekaran:Sneha2036@cluster0.dnmxr1a.mongodb.net/?retryWrites=true&w=majority"
    client = pymongo.MongoClient(CONNECTION_STRING)
    db = client.get_database('SalesDatabase')
    user_collection = pymongo.collection.Collection(db, 'userCollection')
    print("Db connected!..")
except:
    print("Network not connected")
    print("DB not connected!..")

# mongodb+srv://csnehachandrasekaran:Sneha2036@cluster0.dnmxr1a.mongodb.net/?retryWrites=true&w=majority
