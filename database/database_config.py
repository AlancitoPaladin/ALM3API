from flask_pymongo import PyMongo
import os
from dotenv import load_dotenv

load_dotenv()

mongo = PyMongo()


def init_db(app):
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        raise ValueError("La variable de entorno MONGO_URI no está definida.")
    app.config["MONGO_URI"] = mongo_uri
    mongo.init_app(app)