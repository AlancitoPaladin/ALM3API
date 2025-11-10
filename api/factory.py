from flask import Flask

from database.database_config import init_db, mongo
from api.auth_routes import auth_bp


def create_app():
    app = Flask(__name__)

    init_db(app)

    app.register_blueprint(auth_bp)

    return app