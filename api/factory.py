from flask import Flask

from database.database_config import init_db, mongo
from api.auth_routes import auth_bp
from api.model_routes import model_bp


def create_app():
    app = Flask(__name__)

    init_db(app)

    app.register_blueprint(auth_bp)
    app.register_blueprint(model_bp)

    return app