import json
import random
import string

from flask import Blueprint, request, jsonify
from pydantic import ValidationError
from werkzeug.security import generate_password_hash, check_password_hash

from api.utils import send_password
from database.database_config import mongo
from models.user_models import UserModel

auth_bp = Blueprint("auth_bp", __name__)


class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)
        return json.JSONEncoder.default(self, o)


@auth_bp.route('/login', methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}

    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        return jsonify({"message": "Se requieren el email y la contraseña"}), 400

    email = email.strip().lower()

    user_collection = mongo.db.users
    user = user_collection.find_one({"email": email})

    if not user or not check_password_hash(user["password"], password):
        return jsonify({"message": "Correo o contraseña inválida"}), 401

    role = user.get("role")
    if not role:
        return jsonify({"message": "No se pudo obtener el rol del usuario"}), 500

    user_data = {
        "id": str(user["_id"]),
        "email": user["email"],
        "role": role,
        "name": user.get("name", "")
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


@auth_bp.route('/models', methods=['GET'])
def get_models():
    try:
        # Obtener parámetros opcionales
        category = request.args.get('category', None)

        models_collection = mongo.db.models  # Ajusta el nombre de la colección

        # Construir query
        query = {"isActive": True}
        if category:
            query["category"] = category

        # Obtener modelos de la base de datos
        models_cursor = models_collection.find(query)
        models_list = list(models_cursor)

        # Convertir ObjectId a string
        for model in models_list:
            model['_id'] = str(model['_id'])
            # Asegurar que tenga los campos requeridos
            model.setdefault('rating', 0.0)
            model.setdefault('price', 0.0)
            model.setdefault('isActive', True)

        return JSONEncoder().encode({"models": models_list}), 200

    except Exception as e:
        return jsonify({"error": "Error al obtener modelos", "details": str(e)}), 500


@auth_bp.route('/models', methods=['POST'])
def create_model():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No se recibieron datos"}), 400

        # Validar campos requeridos
        required_fields = ['name', 'description', 'imageUrl', 'price', 'category']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Campo requerido faltante: {field}"}), 400

        model_data = {
            "name": data['name'],
            "description": data['description'],
            "imageUrl": data['imageUrl'],
            "rating": data.get('rating', 0.0),
            "price": data['price'],
            "category": data['category'],
            "isActive": data.get('isActive', True)
        }

        models_collection = mongo.db.models
        result = models_collection.insert_one(model_data)

        return jsonify({
            "message": "Modelo creado exitosamente",
            "id": str(result.inserted_id)
        }), 201

    except Exception as e:
        return jsonify({"error": "Error al crear modelo", "details": str(e)}), 500


@auth_bp.route('/seller/profile/<user_id>', methods=['GET'])
def get_seller_profile(user_id):
    try:
        user_collection = mongo.db.users
        user = user_collection.find_one({'_id': ObjectId(user_id)})

        if not user:
            return jsonify({"error": "Usuario no encontrado"}), 404

        # Buscar tienda del vendedor si existe
        store = mongo.db.stores.find_one({'seller_id': ObjectId(user_id)})

        return jsonify({
            "name": user.get("name", ""),
            "email": user["email"],
            "storeName": store.get("name", "Mi Tienda") if store else "Mi Tienda",
            "storeDescription": store.get("description", "") if store else ""
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
