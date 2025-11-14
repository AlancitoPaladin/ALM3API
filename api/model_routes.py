import os
from io import BytesIO

import boto3
import cv2
import numpy as np
import trimesh
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from flask import Blueprint, request, jsonify, send_file
from sklearn.cluster import KMeans
from ultralytics import YOLO
from werkzeug.utils import secure_filename

load_dotenv()

AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME')
S3_MODELS_PREFIX = '3d_models/'
S3_GENERATED_PREFIX = 'generated_models/'

# Cliente S3
s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)

# Blueprint
model_bp = Blueprint('model', __name__)

# Configuración
UPLOAD_FOLDER = 'uploads'
MODELS_FOLDER = 'models_3d'  # Carpeta con tus modelos base .glb
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Cargar modelo YOLO (una sola vez al iniciar)
yolo_model = None


def init_yolo_model():
    """Inicializa el modelo YOLO v11"""
    global yolo_model
    if yolo_model is None:
        # Descarga automática si no existe
        yolo_model = YOLO('yolo11n.pt')  # yolo11n, yolo11s, yolo11m, yolo11l, yolo11x
    return yolo_model


def allowed_file(filename):
    """Verifica extensión de archivo"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_dominant_colors(image_path, n_colors=3):
    """
    Extrae los colores dominantes de una imagen usando K-Means

    Args:
        image_path: Ruta de la imagen
        n_colors: Número de colores dominantes a extraer

    Returns:
        Lista de colores en formato RGB hex
    """
    # Leer imagen
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Reshape para K-Means
    pixels = image.reshape(-1, 3)

    # Aplicar K-Means
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    kmeans.fit(pixels)

    # Obtener colores dominantes
    colors = kmeans.cluster_centers_.astype(int)

    # Convertir a hex
    colors_hex = ['#{:02x}{:02x}{:02x}'.format(r, g, b) for r, g, b in colors]

    # Calcular porcentajes
    labels = kmeans.labels_
    counts = np.bincount(labels)
    percentages = (counts / len(labels) * 100).tolist()

    return [
        {"color": color, "percentage": round(pct, 2)}
        for color, pct in zip(colors_hex, percentages)
    ]


def segment_and_detect(image_path):
    """
    Realiza segmentación con YOLO v11 y extrae características

    Args:
        image_path: Ruta de la imagen

    Returns:
        Dict con objeto detectado, confianza, bbox y características
    """
    model = init_yolo_model()

    # Realizar inferencia
    results = model(image_path, conf=0.25)

    if len(results) == 0 or len(results[0].boxes) == 0:
        return None

    # Obtener mejor detección
    result = results[0]
    boxes = result.boxes

    # Ordenar por confianza y tomar el mejor
    confidences = boxes.conf.cpu().numpy()
    best_idx = np.argmax(confidences)

    best_box = boxes[best_idx]
    class_id = int(best_box.cls.cpu().numpy()[0])
    confidence = float(best_box.conf.cpu().numpy()[0])
    bbox = best_box.xyxy.cpu().numpy()[0].tolist()  # [x1, y1, x2, y2]

    # Nombre de la clase
    class_name = model.names[class_id]

    # Extraer colores dominantes
    colors = extract_dominant_colors(image_path)

    # Calcular dimensiones relativas del bbox
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    bbox_width = bbox[2] - bbox[0]
    bbox_height = bbox[3] - bbox[1]

    return {
        "object": class_name,
        "confidence": round(confidence, 4),
        "bbox": {
            "x1": int(bbox[0]),
            "y1": int(bbox[1]),
            "x2": int(bbox[2]),
            "y2": int(bbox[3]),
            "width_percent": round((bbox_width / w) * 100, 2),
            "height_percent": round((bbox_height / h) * 100, 2)
        },
        "colors": colors
    }


def modify_3d_model(base_model_data, colors, scale_factor=1.0):
    """
    Modifica modelo 3D base aplicando colores y escala

    Args:
        base_model_path: Ruta del modelo base. Glb
        colors: Lista de colores dominantes
        scale_factor: Factor de escala (1.0 = sin cambio)

    Returns:
        BytesIO con el modelo modificado
    """
    # Cargar modelo base
    mesh = trimesh.load(base_model_path, force='mesh')

    # Aplicar escala si es necesario
    if scale_factor != 1.0:
        mesh.apply_scale(scale_factor)

    # Aplicar color dominante al modelo
    if colors and len(colors) > 0:
        # Convertir hex a RGB normalizado (0-1)
        primary_color = colors[0]["color"]
        r = int(primary_color[1:3], 16) / 255.0
        g = int(primary_color[3:5], 16) / 255.0
        b = int(primary_color[5:7], 16) / 255.0

        # Crear color RGBA
        color_rgba = np.array([r, g, b, 1.0])

        # Aplicar color a todos los vértices
        mesh.visual.vertex_colors = np.tile(color_rgba, (len(mesh.vertices), 1))

    # Exportar a BytesIO
    output = BytesIO()
    mesh.export(output, file_type='glb')
    output.seek(0)

    return output


def download_model_from_s3(object_key):
    """Descarga modelo 3D desde S3 a memoria"""
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=object_key)
        model_data = BytesIO(response['Body'].read())
        model_data.seek(0)
        return model_data
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            return None
        raise


def upload_model_to_s3(file_data, object_key):
    """Sube modelo 3D modificado a S3"""
    try:
        file_data.seek(0)
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=object_key,
            Body=file_data.getvalue(),
            ContentType='model/gltf-binary'
        )
        url = f"https://{S3_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{object_key}"
        return url
    except ClientError as e:
        raise Exception(f"Error subiendo a S3: {str(e)}")


def list_available_models_in_s3():
    try:
        response = s3_client.list_objects_v2(
            Bucket=S3_BUCKET_NAME,
            Prefix=S3_MODELS_PREFIX
        )
        if 'Contents' not in response:
            return []

        models = []
        for obj in response['Contents']:
            key = obj['Key']
            if key.endswith('.glb'):
                model_name = key.replace(S3_MODELS_PREFIX, '').replace('.glb', '')
                models.append(model_name)
        return models
    except ClientError as e:
        print(f"Error listando modelos: {str(e)}")
        return []


@model_bp.route('/process-image', methods=['POST'])
def process_image():
    """
    Endpoint para procesar imagen y generar modelo 3D

    Expected: multipart/form-data con campo 'image'
    Returns: JSON con datos de detección y URL del modelo 3D
    """
    # Validar que hay archivo
    if 'image' not in request.files:
        return jsonify({"message": "No se envió ninguna imagen"}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({"message": "Nombre de archivo vacío"}), 400

    if not allowed_file(file.filename):
        return jsonify({"message": "Formato de imagen no permitido"}), 400

    try:
        # Guardar imagen temporalmente
        filename = secure_filename(file.filename)
        timestamp = str(int(np.random.random() * 1000000))
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(filepath)

        # Procesar con YOLO y extraer características
        detection_data = segment_and_detect(filepath)

        if not detection_data:
            os.remove(filepath)
            return jsonify({"message": "No se detectó ningún objeto en la imagen"}), 404

        # Mapear objeto detectado a modelo 3D en S3
        detected_object = detection_data["object"]
        s3_model_key = f"{S3_MODELS_PREFIX}{detected_object}.glb"

        # Descargar modelo base desde S3
        base_model_data = download_model_from_s3(s3_model_key)

        # Si no existe, intentar con default
        if not base_model_data:
            s3_model_key = f"{S3_MODELS_PREFIX}default.glb"
            base_model_data = download_model_from_s3(s3_model_key)

            if not base_model_data:
                os.remove(filepath)
                return jsonify({
                    "message": f"No se encontró modelo 3D para '{detected_object}' en S3",
                    "detection": detection_data,
                    "available_models": list_available_models_in_s3()
                }), 404

        # Modificar modelo 3D con colores detectados
        colors = detection_data["colors"]
        modified_model = modify_3d_model(base_model_data, colors)

        # Subir modelo modificado a S3
        output_filename = f"{timestamp}_{detected_object}.glb"
        s3_output_key = f"{S3_GENERATED_PREFIX}{output_filename}"

        model_url = upload_model_to_s3(modified_model, s3_output_key)

        # Limpiar imagen original
        os.remove(filepath)

        return jsonify({
            "success": True,
            "message": "Imagen procesada exitosamente",
            "detection": detection_data,
            "model_url": model_url,
            "model_key": s3_output_key,
            "model_filename": output_filename
        }), 200

    except Exception as e:
        # Limpiar archivos en caso de error
        if os.path.exists(filepath):
            os.remove(filepath)

        return jsonify({
            "message": "Error al procesar la imagen",
            "error": str(e)
        }), 500


@model_bp.route('/available-models', methods=['GET'])
def get_available_models():
    """Endpoint para listar modelos base disponibles en S3"""
    try:
        models = list_available_models_in_s3()
        return jsonify({
            "success": True,
            "models": models,
            "count": len(models)
        }), 200
    except Exception as e:
        return jsonify({
            "message": "Error al listar modelos",
            "error": str(e)
        }), 500


@model_bp.route('/download/<filename>', methods=['GET'])
def download_model(filename):
    """
    Endpoint para descargar modelo 3D generado
    """
    try:
        filepath = os.path.join(UPLOAD_FOLDER, secure_filename(filename))

        if not os.path.exists(filepath):
            return jsonify({"message": "Archivo no encontrado"}), 404

        return send_file(
            filepath,
            mimetype='model/gltf-binary',
            as_attachment=True,
            download_name=filename
        )

    except Exception as e:
        return jsonify({"message": "Error al descargar archivo", "error": str(e)}), 500
