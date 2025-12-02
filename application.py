from api.factory import create_app
from database.database_config import init_db

application = create_app()

application.config["DEBUG"] = True

init_db(application)

if __name__ == "__main__":
    application.run(debug=True, host='0.0.0.0', port=5000)
