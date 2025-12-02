from api.factory import create_app
from database.database_config import init_db

app = create_app()

app.config["DEBUG"] = True

init_db(app)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
