import os
from flask import Flask
from api.factory import create_app
from database.database_config import init_db, mongo

app = create_app()

app.config["DEBUG"] = True

init_db(app)

if __name__ == "__main__":
    app.run()
