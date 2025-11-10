from flask import Blueprint, request, jsonify
from bson import ObjectId
from database.database_config import mongo
from models.user_models import UserModel
from werkzeug.security import generate_password_hash, check_password_hash
from api.utils import send_password
from pydantic import ValidationError
import random
import string

auth_bp = Blueprint("auth_bp", __name__)


@auth_bp.route('/login', methods=["POST"])
def login():
    data = request.get_json()

    if not data:
        return jsonify({"message": "Solicitud inválida"}), 400

    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        return jsonify({"message": "Se requieren el email y la contraseña"}), 400

    user_collection = mongo.db.users
    user = user_collection.find_one({"email": email})

    if not user or not check_password_hash(user["password"], password):
        return jsonify({"message": "Correo o contraseña inválida"}), 401

    role = user.get("role", None)
    if not role:
        return jsonify({"message": "No se pudo obtener el rol del usuario"}), 500

    user_data = {
        "id": str(user["_id"]),
        "email": user["email"],
        "role": role
    }

    return jsonify({"message": "Ingreso completado", "user": user_data}), 200


@auth_bp.route('/register', methods=["POST"])
def register():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No se recibieron datos en formato JSON"}), 400

        user_data = UserModel(**data)

        user_collection = mongo.db.users

        if user_collection.find_one({'email': user_data.email}):
            return jsonify({"error": "Correo ya registrado"}), 409

        hashed_password = generate_password_hash(user_data.password)

        user_dict = user_data.model_dump()
        user_dict["password"] = hashed_password

        user_collection.insert_one(user_dict)

        return jsonify({"message": "Usuario registrado"}), 201

    except ValidationError as e:
        return jsonify({"error": "Datos inválidos", "details": e.errors()}), 400
    except Exception as e:
        return jsonify({"error": "Error interno del servidor", "details": str(e)}), 500


@auth_bp.route('/reset_password', methods=["POST"])
def reset_password():
    data = request.get_json()
    if not data or "email" not in data:
        return jsonify({"error": "No se recibió el correo"}), 400

    email = data["email"]
    user_collection = mongo.db.users
    user = user_collection.find_one({'email': email})

    if not user:
        return jsonify({"error": "Correo no registrado"}), 404

    new_password = ''.join(random.choices(string.ascii_letters + string.digits + "!@#$%^&*", k=10))
    hashed_password = generate_password_hash(new_password)

    user_collection.update_one(
        {'email': email},
        {'$set': {'password': hashed_password}}
    )

    send_password(email, new_password)

    return jsonify({"message": "Contraseña restablecida y enviada al correo"}), 200


@auth_bp.route('/notifications', methods=["POST"])
def notifications():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No se recibieron datos"}), 400

    user_collection = mongo.db.users

    if user_collection.find_one({'email': data['email']}):
        send_notification()
        return jsonify({"message": "Correo realizado"}), 200
    return None