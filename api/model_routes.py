import hashlib
import logging
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from contextlib import contextmanager
from datetime import datetime, timedelta
from io import BytesIO
from threading import Thread, RLock, Event

import boto3
import cv2
import numpy as np
import trimesh
from botocore.config import Config
from botocore.exceptions import ClientError
from botocore.exceptions import ReadTimeoutError
from bson import ObjectId
from dotenv import load_dotenv
from flask import Blueprint, request, jsonify
from flask import send_file
from sklearn.cluster import KMeans
from trimesh.util import concatenate
from ultralytics import YOLO
from werkzeug.utils import secure_filename

from database.database_config import mongo

model_bp = Blueprint("model_bp", __name__)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | [%(threadName)-12s] | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

load_dotenv()

AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME')
S3_MODELS_PREFIX = '3d_models/'
S3_GENERATED_PREFIX = 'generated_models/'

s3_config = Config(
    connect_timeout=30,
    read_timeout=120,
    retries={
        'max_attempts': 3,
        'mode': 'adaptive'
    }
)

# Cliente S3
s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION,
    config=s3_config
)

TRIMESH_EXECUTOR_MAX_WORKERS = 2
TRIMESH_LOAD_TIMEOUT = 30  # segundos

TASK_MAX_DURATION_MINUTES = 5
TASK_EXPIRY_HOURS = 24
MAX_TASK_HISTORY = 500

UPLOAD_FOLDER = "./uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

yolo_model = None
processing_status = {}

# Executor para operaciones no-bloqueantes
trimesh_executor = ThreadPoolExecutor(max_workers=TRIMESH_EXECUTOR_MAX_WORKERS)

# Caché para detecciones con timestamp de expiración
detection_cache = {}
CACHE_EXPIRY_HOURS = 24
MAX_CACHE_SIZE = 1000

# Cache de modelos 3D
model_cache = {}


# ============ PROTECCIÓN THREAD-SAFE ============

class ThreadSafeTaskManager:
    """Gestor thread-safe para tareas con protección contra deadlocks"""

    def __init__(self):
        self._lock = RLock()
        self._tasks = {}
        self._timestamps = {}
        self._cancellation_flags = {}  # Flags para cancelar tareas

    def update_status(self, task_id, state, progress, step, **kwargs):
        """Actualiza estado de forma thread-safe"""
        with self._lock:
            current_state = self._tasks.get(task_id, {}).get("state")

            if current_state in ("completed", "failed", "cancelled"):
                logger.warning(
                    f"[{task_id}] Intentando cambiar desde estado terminal: {current_state} -> {state}"
                )
                return

            self._tasks[task_id] = {
                "state": state,  # uso interno backend
                "status": state,  # ← lo que consume Android

                "progress": max(0, min(100, progress)),
                "current_step": step,
                "timestamp": datetime.now().isoformat(),
                "start_time": self._timestamps.get(task_id),
                "timings": self._tasks.get(task_id, {}).get("timings", {}),
                **kwargs
            }

            logger.info(f"[{task_id}] → {state} | {progress}% | {step}")

    def get_status(self, task_id):
        """Obtiene estado de forma thread-safe"""
        with self._lock:
            return self._tasks.get(task_id)

    def exists(self, task_id):
        """Verifica si tarea existe"""
        with self._lock:
            return task_id in self._tasks

    def init_task(self, task_id):
        """Inicializa tarea con timestamp de inicio"""
        with self._lock:
            self._timestamps[task_id] = datetime.now().isoformat()
            self._tasks[task_id] = {
                "state": "pending",
                "progress": 0,
                "current_step": "Inicializando...",
                "timestamp": datetime.now().isoformat(),
                "start_time": self._timestamps[task_id],
                "timings": {},
                "watchdog_count": 0
            }
            self._cancellation_flags[task_id] = Event()
            logger.info(f"[{task_id}] Tarea inicializada")

    def record_timing(self, task_id, stage_name, elapsed_seconds, details=None):
        """Registra tiempo de etapa con detalles opcionales"""
        with self._lock:
            if task_id in self._tasks:
                if "timings" not in self._tasks[task_id]:
                    self._tasks[task_id]["timings"] = {}

                self._tasks[task_id]["timings"][stage_name] = {
                    "seconds": round(elapsed_seconds, 3),
                    "timestamp": datetime.now().isoformat(),
                    "details": details or {}
                }
                logger.debug(f"[{task_id}] Timing registrado: {stage_name} = {elapsed_seconds:.3f}s")

    def get_all_tasks(self):
        """Obtiene todas las tareas"""
        with self._lock:
            return dict(self._tasks)

    def get_timings(self, task_id):
        """Obtiene timings de tarea"""
        with self._lock:
            if task_id in self._tasks:
                return self._tasks[task_id].get("timings", {})
            return {}

    def check_timeout(self, task_id):
        """Verifica si tarea excedió timeout"""
        with self._lock:
            if task_id not in self._timestamps:
                return False

            start_time = datetime.fromisoformat(self._timestamps[task_id])
            elapsed = (datetime.now() - start_time).total_seconds() / 60

            if elapsed > TASK_MAX_DURATION_MINUTES:
                logger.warning(f"[{task_id}] TIMEOUT: {elapsed:.1f}m > {TASK_MAX_DURATION_MINUTES}m")
                return True

            return False

    def increment_watchdog(self, task_id):
        """Incrementa contador watchdog para detectar deadlocks"""
        with self._lock:
            if task_id in self._tasks:
                current = self._tasks[task_id].get("watchdog_count", 0)
                self._tasks[task_id]["watchdog_count"] = current + 1
                return current + 1
            return 0

    def should_cancel(self, task_id):
        """Verifica si tarea debe ser cancelada"""
        with self._lock:
            if task_id not in self._cancellation_flags:
                return False
            return self._cancellation_flags[task_id].is_set()

    def cancel_task(self, task_id):
        """Marca tarea para cancelación"""
        with self._lock:
            if task_id in self._cancellation_flags:
                self._cancellation_flags[task_id].set()
                logger.info(f"[{task_id}] Cancelación solicitada")

    def cleanup_expired(self):
        """Limpia tareas expiradas y detecta stuck tasks"""
        with self._lock:
            expired_tasks = []
            stuck_tasks = []

            cutoff_time = datetime.now() - timedelta(hours=TASK_EXPIRY_HOURS)

            for task_id, status in list(self._tasks.items()):
                try:
                    task_time = datetime.fromisoformat(status.get('timestamp', ''))
                    start_time = datetime.fromisoformat(status.get('start_time', ''))

                    # Detectar tareas stuck en processing
                    elapsed_mins = (datetime.now() - start_time).total_seconds() / 60
                    if status.get("state") == "processing" and elapsed_mins > TASK_MAX_DURATION_MINUTES:
                        stuck_tasks.append((task_id, elapsed_mins))
                        self._tasks[task_id]["state"] = "timeout"
                        self._tasks[task_id]["current_step"] = f"TIMEOUT después de {elapsed_mins:.1f}m"

                    # Limpiar completadas/fallidas/canceladas viejas
                    if task_time < cutoff_time and status.get("state") in ["completed", "failed", "cancelled",
                                                                           "timeout"]:
                        expired_tasks.append(task_id)

                except (ValueError, KeyError):
                    expired_tasks.append(task_id)

            # Limpiar
            for task_id in expired_tasks:
                del self._tasks[task_id]
                self._timestamps.pop(task_id, None)
                self._cancellation_flags.pop(task_id, None)

            if expired_tasks:
                logger.info(f" Limpiadas {len(expired_tasks)} tareas expiradas")

            if stuck_tasks:
                logger.warning(f" Detectadas {len(stuck_tasks)} tareas stuck:")
                for task_id, elapsed in stuck_tasks:
                    logger.warning(f"  - {task_id}: {elapsed:.1f}m")

            # Limpieza si hay demasiadas tareas
            if len(self._tasks) > MAX_TASK_HISTORY:
                oldest = sorted(
                    self._tasks.items(),
                    key=lambda x: x[1].get('timestamp', '')
                )[:len(self._tasks) - MAX_TASK_HISTORY]

                for task_id, _ in oldest:
                    del self._tasks[task_id]
                    self._timestamps.pop(task_id, None)
                    self._cancellation_flags.pop(task_id, None)

                logger.info(f" Limpiadas {len(oldest)} tareas antiguas por límite histórico")

    def get_diagnostics(self):
        """Obtiene diagnósticos del sistema"""
        with self._lock:
            states_count = {}
            stuck_count = 0

            for task_id, status in self._tasks.items():
                state = status.get('state', 'unknown')
                states_count[state] = states_count.get(state, 0) + 1

                if state == "processing":
                    start_time = datetime.fromisoformat(status.get('start_time', datetime.now().isoformat()))
                    elapsed_mins = (datetime.now() - start_time).total_seconds() / 60
                    if elapsed_mins > TASK_MAX_DURATION_MINUTES * 0.8:  # 80% del timeout
                        stuck_count += 1

            return {
                "total_tasks": len(self._tasks),
                "tasks_by_state": states_count,
                "potential_stuck_tasks": stuck_count,
                "max_task_duration_minutes": TASK_MAX_DURATION_MINUTES
            }


# Instancia global thread-safe
task_manager = ThreadSafeTaskManager()


# Caché thread-safe
class ThreadSafeCache:
    """Cache thread-safe con expiración"""

    def __init__(self, max_size=1000, expiry_hours=24):
        self._lock = RLock()
        self._cache = {}
        self._metadata = {}
        self.max_size = max_size
        self.expiry_hours = expiry_hours

    def get(self, key):
        """Obtiene valor del caché"""
        with self._lock:
            if key not in self._cache:
                return None

            entry = self._cache[key]
            if not self._is_valid(entry):
                del self._cache[key]
                self._metadata.pop(key, None)
                return None

            return entry.get('data')

    def set(self, key, value, metadata=None):
        """Guarda valor en caché"""
        with self._lock:
            if len(self._cache) >= self.max_size:
                self._cleanup_oldest()

            self._cache[key] = {
                'data': value,
                'timestamp': datetime.now().isoformat()
            }

            if metadata:
                self._metadata[key] = metadata

    def _is_valid(self, entry):
        """Verifica validez de entrada"""
        try:
            timestamp = datetime.fromisoformat(entry.get('timestamp', ''))
            expiry = timestamp + timedelta(hours=self.expiry_hours)
            return datetime.now() < expiry
        except (ValueError, TypeError):
            return False

    def _cleanup_oldest(self):
        """Elimina entradas más antiguas"""
        oldest_keys = sorted(
            self._cache.items(),
            key=lambda x: x[1].get('timestamp', ''),
            reverse=True
        )[self.max_size - 10:]

        for key, _ in oldest_keys:
            del self._cache[key]
            self._metadata.pop(key, None)

    def size(self):
        """Obtiene tamaño del caché"""
        with self._lock:
            return len(self._cache)

    def clear(self):
        """Limpia caché"""
        with self._lock:
            self._cache.clear()
            self._metadata.clear()

    def get_metadata(self, key):
        """Obtiene metadatos"""
        with self._lock:
            return self._metadata.get(key)


# Instancias de caché
detection_cache = ThreadSafeCache(max_size=1000, expiry_hours=24)
model_cache = ThreadSafeCache(max_size=100, expiry_hours=24)


# ============ UTILIDADES - CONTEXT MANAGER ============

@contextmanager
def timed_stage(stage_name, task_id, expected_duration_s=None):
    """
    Context manager para registrar tiempo de etapas de procesamiento

    Args:
        stage_name: Nombre de la etapa
        task_id: ID de la tarea
        expected_duration_s: Duración esperada en segundos (opcional)
    """
    start_time = time.time()
    logger.debug(f"[{task_id}]  Iniciando: {stage_name}")

    try:
        yield
    finally:
        elapsed = time.time() - start_time
        task_manager.record_timing(task_id, stage_name, elapsed, {
            "expected_seconds": expected_duration_s
        })

        status_msg = f"[{task_id}] {stage_name}: {elapsed:.3f}s"
        if expected_duration_s and elapsed > expected_duration_s * 1.5:
            status_msg += f" (esperado: ~{expected_duration_s}s)"
        logger.debug(status_msg)


# ============ TRIMESH NO-BLOQUEANTE ============


def load_mesh_non_blocking(data, file_type='glb', task_id=None):
    """
    Carga mesh de forma no-bloqueante usando Scene y conversión

    Returns:
        (mesh, mesh_info) o (None, None) si falla
    """

    def _load_internal():
        start = time.time()

        try:
            logger.info(f"[{task_id}] Leyendo archivo {file_type}...")
            data.seek(0)

            # Usar load_mesh que es más seguro que load()
            loaded = trimesh.load_mesh(data, file_type=file_type, process=False)
            elapsed = time.time() - start

            # Ya es Mesh, validar
            if isinstance(loaded, trimesh.Trimesh):
                logger.info(f"[{task_id}] Mesh cargado directamente en {elapsed:.3f}s")
                mesh = loaded

            else:
                logger.error(f"[{task_id}] Tipo inesperado: {type(loaded)}")
                return None, None

            # Validaciones
            if len(mesh.vertices) == 0:
                raise ValueError("Mesh sin vértices")
            if len(mesh.faces) == 0:
                raise ValueError("Mesh sin caras")

            # Usar atributos disponibles en trimesh
            is_valid = mesh.is_valid if hasattr(mesh, 'is_valid') else True
            is_watertight = mesh.is_watertight if hasattr(mesh, 'is_watertight') else False

            mesh_info = {
                "vertices": len(mesh.vertices),
                "faces": len(mesh.faces),
                "is_valid": is_valid,
                "is_watertight": is_watertight,
                "bounds_size": [round(float(x), 2) for x in (mesh.bounds[1] - mesh.bounds[0])],
                "load_time_seconds": round(elapsed, 3)
            }

            logger.info(f"[{task_id}] Mesh válido: {mesh_info['vertices']} verts, {mesh_info['faces']} faces")

            return mesh, mesh_info

        except Exception as e:
            elapsed = time.time() - start
            logger.error(f"[{task_id}] Error cargando mesh ({elapsed:.3f}s): {str(e)}")
            return None, None

    # Ejecutar en thread pool con timeout
    try:
        future = trimesh_executor.submit(_load_internal)
        mesh, info = future.result(timeout=TRIMESH_LOAD_TIMEOUT)
        return mesh, info

    except FuturesTimeoutError:
        logger.error(f"[{task_id}] TIMEOUT cargando mesh ({TRIMESH_LOAD_TIMEOUT}s)")
        return None, None
    except Exception as e:
        logger.error(f"[{task_id}] Error en executor: {str(e)}")
        return None, None


def convert_scene_to_mesh(scene_data, file_type='glb', task_id=None):
    """
    Convierte Scene de Trimesh a Mesh combinado

    Returns:
        (mesh, mesh_info) o (None, None)
    """

    def _convert_internal():
        start = time.time()

        try:
            logger.info(f"[{task_id}] Cargando Scene {file_type}...")
            scene_data.seek(0)

            # Cargar como Scene
            scene = trimesh.load(scene_data, file_type=file_type, process=False)

            if isinstance(scene, trimesh.Trimesh):
                # Ya es mesh
                logger.info(f"[{task_id}] Objeto es Mesh, no Scene")
                return scene, {"load_type": "direct_mesh"}

            if isinstance(scene, trimesh.Scene):
                logger.info(f"[{task_id}] Scene cargada con {len(scene.geometry)} geometrías")

                if len(scene.geometry) == 0:
                    raise ValueError("Scene vacía")

                # Convertir a mesh único
                meshes = []
                for key, geom in scene.geometry.items():
                    if isinstance(geom, trimesh.Trimesh):
                        meshes.append(geom)
                        logger.debug(f"[{task_id}]   + Mesh '{key}': {len(geom.vertices)} verts")

                if not meshes:
                    raise ValueError("Scene sin mallas válidas")

                # Combinar
                logger.info(f"[{task_id}] Combinando {len(meshes)} mallas...")
                mesh = concatenate(meshes)

                elapsed = time.time() - start
                mesh_info = {
                    "vertices": len(mesh.vertices),
                    "faces": len(mesh.faces),
                    "is_valid": mesh.is_valid,
                    "meshes_combined": len(meshes),
                    "load_type": "scene_converted",
                    "load_time_seconds": round(elapsed, 3)
                }

                logger.info(f"[{task_id}] Scene convertido: {mesh_info}")
                return mesh, mesh_info

            else:
                raise ValueError(f"Tipo no soportado: {type(scene)}")

        except Exception as e:
            elapsed = time.time() - start
            logger.error(f"[{task_id}] Error convertiendo ({elapsed:.3f}s): {str(e)}")
            return None, None

    # Ejecutar con timeout
    try:
        future = trimesh_executor.submit(_convert_internal)
        mesh, info = future.result(timeout=TRIMESH_LOAD_TIMEOUT)
        return mesh, info
    except FuturesTimeoutError:
        logger.error(f"[{task_id}] TIMEOUT en conversión ({TRIMESH_LOAD_TIMEOUT}s)")
        return None, None
    except Exception as e:
        logger.error(f"[{task_id}] Error en executor: {str(e)}")
        return None, None


def compute_image_hash(image_path):
    """
    Computa hash MD5 de la imagen para caché

    Args:
        image_path: Ruta a la imagen

    Returns:
        String del hash MD5
    """
    hash_md5 = hashlib.md5()
    with open(image_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def is_cache_valid(cache_entry):
    """Verifica si una entrada de caché sigue siendo válida"""
    if 'timestamp' not in cache_entry:
        return False

    expiry_time = datetime.fromisoformat(cache_entry['timestamp']) + timedelta(hours=CACHE_EXPIRY_HOURS)
    return datetime.now() < expiry_time


def get_cached_detection(image_hash):
    """
    Obtiene detección en caché si existe y es válida

    Args:
        image_hash: Hash de la imagen

    Returns:
        Datos de detección o None
    """
    if image_hash in detection_cache:
        cache_entry = detection_cache[image_hash]
        if is_cache_valid(cache_entry):
            return cache_entry['data']
        else:
            # Limpiar entrada expirada
            del detection_cache[image_hash]

    return None


def cache_detection(image_hash, detection_data):
    """Guarda detección en caché con límite de tamaño"""
    # Limpiar caché si es muy grande
    if len(detection_cache) >= MAX_CACHE_SIZE:
        # Remover las 10 entradas más antiguas
        oldest_keys = sorted(
            detection_cache.items(),
            key=lambda x: x[1].get('timestamp', ''),
            reverse=True
        )[MAX_CACHE_SIZE - 10:]
        for key, _ in oldest_keys:
            del detection_cache[key]

    detection_cache[image_hash] = {
        'data': detection_data,
        'timestamp': datetime.now().isoformat()
    }


def crop_image_by_bbox(image_path, bbox):
    """
    Recorta la imagen usando el bounding box detectado

    Args:
        image_path: Ruta a la imagen original
        bbox: Dict con claves x1, y1, x2, y2

    Returns:
        numpy array con imagen recortada
    """
    image = cv2.imread(image_path)

    # Agregar padding para evitar bordes
    padding = 10
    x1 = max(0, bbox['x1'] - padding)
    y1 = max(0, bbox['y1'] - padding)
    x2 = min(image.shape[1], bbox['x2'] + padding)
    y2 = min(image.shape[0], bbox['y2'] + padding)

    cropped = image[int(y1):int(y2), int(x1):int(x2)]

    return cropped


def extract_dominant_colors(image_input, n_colors=3, is_cropped=False, task_id=None):
    """
    Extrae colores dominantes (optimizado para imagen recortada)

    Args:
        image_input: Ruta (str) o array numpy de la imagen
        n_colors: Número de colores a extraer
        is_cropped: Si la imagen ya está recortada
        task_id: ID de la tarea (para logging)

    Returns:
        Lista de colores con porcentajes
    """
    # Soportar tanto rutas como arrays
    if isinstance(image_input, str):
        image = cv2.imread(image_input)
    else:
        image = image_input.copy()

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Redimensionar solo si es necesario (imagen grande)
    height, width = image.shape[:2]
    target_size = 100 if is_cropped else 150

    if height > target_size or width > target_size:
        image = cv2.resize(image, (target_size, target_size))

    pixels = image.reshape(-1, 3)

    # Usar solo 3 iteraciones para imágenes recortadas (más rápido)
    n_init = 3 if is_cropped else 5
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=n_init)
    kmeans.fit(pixels)

    colors = kmeans.cluster_centers_.astype(int)
    colors_hex = ['#{:02x}{:02x}{:02x}'.format(r, g, b) for r, g, b in colors]

    labels = kmeans.labels_
    counts = np.bincount(labels)
    percentages = (counts / len(labels) * 100).tolist()

    result = [
        {"color": color, "percentage": round(pct, 2)}
        for color, pct in zip(colors_hex, percentages)
    ]

    logger.debug(f"[{task_id}] Colores extraídos: {len(result)} colores")

    return result


def segment_and_detect(image_path, use_cache=True, task_id=None):
    """Detección YOLO con mejor manejo de errores"""
    try:
        logger.info(f"[{task_id}] Verificando si archivo existe: {os.path.exists(image_path)}")

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Imagen no existe: {image_path}")

        if use_cache:
            try:
                image_hash = compute_image_hash(image_path)
                cached_result = detection_cache.get(image_hash)
                if cached_result is not None:
                    logger.info(f"[{task_id}] Detección en caché")
                    return cached_result
            except Exception as e:
                logger.warning(f"[{task_id}] No se pudo usar caché: {str(e)}")

        with timed_stage("Inferencia YOLO", task_id, expected_duration_s=5):
            logger.info(f"[{task_id}] Cargando modelo YOLO")
            model = init_yolo_model()
            logger.info(f"[{task_id}] Ejecutando inferencia")
            results = model(image_path, conf=0.5, imgsz=416)
            logger.info(f"[{task_id}] Inferencia completada, resultados: {len(results)}")

        if len(results) == 0 or len(results[0].boxes) == 0:
            logger.warning(f"[{task_id}] Sin detecciones en resultados")
            return None

        logger.info(f"[{task_id}] Procesando resultados")
        result = results[0]
        boxes = result.boxes
        confidences = boxes.conf.cpu().numpy()
        best_idx = np.argmax(confidences)

        best_box = boxes[best_idx]
        class_id = int(best_box.cls.cpu().numpy()[0])
        confidence = float(best_box.conf.cpu().numpy()[0])
        bbox = best_box.xyxy.cpu().numpy()[0].tolist()

        class_name = model.names[class_id]
        logger.info(f"[{task_id}] Objeto detectado: {class_name} (confianza: {confidence:.2f})")

        bbox_dict = {
            "x1": int(bbox[0]),
            "y1": int(bbox[1]),
            "x2": int(bbox[2]),
            "y2": int(bbox[3])
        }

        with timed_stage("Procesamiento ROI", task_id):
            cropped_image = crop_image_by_bbox(image_path, bbox_dict)
            colors = extract_dominant_colors(cropped_image, is_cropped=True, task_id=task_id)

        image = cv2.imread(image_path)
        h, w = image.shape[:2]
        bbox_width = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]

        detection_data = {
            "object": class_name,
            "confidence": round(confidence, 4),
            "bbox": {
                **bbox_dict,
                "width_percent": round((bbox_width / w) * 100, 2),
                "height_percent": round((bbox_height / h) * 100, 2)
            },
            "colors": colors
        }

        if use_cache:
            try:
                detection_cache.set(image_hash, detection_data)
                logger.info(f"[{task_id}] Detección cacheada")
            except Exception as e:
                logger.warning(f"[{task_id}] No se pudo cachear: {str(e)}")

        logger.info(f"[{task_id}] Detección exitosa")
        return detection_data

    except FileNotFoundError as e:
        logger.error(f"[{task_id}] Archivo no encontrado: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"[{task_id}] Error en detección: {str(e)}\n{traceback.format_exc()}")
        raise


def init_yolo_model():
    """Inicializa el modelo YOLO v11 con optimizaciones"""
    global yolo_model
    if yolo_model is None:
        yolo_model = YOLO('yolo11n.pt')
        # Usar CPU (cambiar a 'cuda' si tienes GPU disponible)
        yolo_model.to('cpu')
    return yolo_model


def allowed_file(filename):
    """Verifica extensión de archivo"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def process_image_background(filepath, unique_filename, user_id=None):
    """Procesa imagen con protección contra processing infinito"""
    task_id = unique_filename

    try:
        task_manager.init_task(task_id)
        task_manager.cleanup_expired()

        logger.info(f"[{task_id}] ======== INICIO ========")
        logger.info(f"[{task_id}] User ID: {user_id}")
        logger.info(f"[{task_id}] Ruta archivo: {filepath}")

        # ==================== DETECCIÓN ====================
        if task_manager.should_cancel(task_id):
            raise Exception("Cancelada antes de iniciar")

        if task_manager.check_timeout(task_id):
            raise TimeoutError(f"Timeout total ({TASK_MAX_DURATION_MINUTES}m)")

        task_manager.update_status(task_id, "yolo_detecting", 10, "Detectando...")
        logger.info(f"[{task_id}] Iniciando detección YOLO")

        try:
            detection_data = segment_and_detect(filepath, task_id=task_id)
            logger.info(f"[{task_id}] Detección completada: {detection_data is not None}")
        except Exception as e:
            logger.error(f"[{task_id}] Excepción en detección: {str(e)}\n{traceback.format_exc()}")
            task_manager.update_status(task_id, "failed", 0, f"Detección: {str(e)[:100]}")
            if os.path.exists(filepath):
                os.remove(filepath)
            return

        if not detection_data:
            logger.warning(f"[{task_id}] Sin datos de detección")
            task_manager.update_status(task_id, "failed", 0,
                                       "Sin detecciones - No se detectó ningún objeto en la imagen")
            if os.path.exists(filepath):
                os.remove(filepath)
            return

        logger.info(f"[{task_id}] Objeto detectado: {detection_data.get('object', 'unknown')}")

        # ==================== CARGAR MODELO ====================
        if task_manager.should_cancel(task_id):
            task_manager.update_status(task_id, "cancelled", 0, "Cancelada por usuario")
            if os.path.exists(filepath):
                os.remove(filepath)
            return

        if task_manager.check_timeout(task_id):
            task_manager.update_status(task_id, "timeout", 0, f"Timeout total ({TASK_MAX_DURATION_MINUTES}m)")
            if os.path.exists(filepath):
                os.remove(filepath)
            return

        task_manager.update_status(task_id, "loading_model", 25, "Descargando modelo...")
        logger.info(f"[{task_id}] Iniciando descarga de modelo")

        detected_object = detection_data["object"]
        s3_model_key = f"{S3_MODELS_PREFIX}{detected_object}.glb"
        logger.info(f"[{task_id}] Buscando: {s3_model_key}")

        base_model_data = download_model_from_s3(s3_model_key, task_id=task_id)

        if not base_model_data:
            logger.info(f"[{task_id}] Modelo específico no encontrado, usando default")
            s3_model_key = f"{S3_MODELS_PREFIX}default.glb"
            base_model_data = download_model_from_s3(s3_model_key, task_id=task_id)

            if not base_model_data:
                logger.error(f"[{task_id}] Ni modelo específico ni default encontrados")
                task_manager.update_status(task_id, "failed", 0,
                                           f"Modelo no encontrado - No hay modelo para '{detected_object}' ni default.glb")
                if os.path.exists(filepath):
                    os.remove(filepath)
                return

        logger.info(f"[{task_id}] Modelo descargado exitosamente")

        # ==================== EXTRAER COLORES ====================
        if task_manager.should_cancel(task_id):
            task_manager.update_status(task_id, "cancelled", 0, "Cancelada por usuario")
            if os.path.exists(filepath):
                os.remove(filepath)
            return

        task_manager.update_status(task_id, "extracting_colors", 45, "Extrayendo colores...")
        colors = detection_data["colors"]
        logger.info(f"[{task_id}] Colores extraídos: {len(colors)} colores")

        # ==================== MODIFICAR MODELO ====================
        if task_manager.should_cancel(task_id):
            task_manager.update_status(task_id, "cancelled", 0, "Cancelada por usuario")
            if os.path.exists(filepath):
                os.remove(filepath)
            return

        if task_manager.check_timeout(task_id):
            task_manager.update_status(task_id, "timeout", 0, f"Timeout total ({TASK_MAX_DURATION_MINUTES}m)")
            if os.path.exists(filepath):
                os.remove(filepath)
            return

        task_manager.update_status(task_id, "modifying_model", 60, "Aplicando colores...")
        logger.info(f"[{task_id}] Iniciando modificación de modelo")

        try:
            modified_model = modify_3d_model(base_model_data, colors, task_id=task_id)
            logger.info(f"[{task_id}] Modelo modificado exitosamente")
        except Exception as e:
            logger.error(f"[{task_id}] Excepción en modificación: {str(e)}\n{traceback.format_exc()}")
            task_manager.update_status(task_id, "failed", 0, f"Error modificando modelo: {str(e)[:100]}")
            if os.path.exists(filepath):
                os.remove(filepath)
            return

        # SUBIR A S3
        if task_manager.should_cancel(task_id):
            task_manager.update_status(task_id, "cancelled", 0, "Cancelada por usuario")
            if os.path.exists(filepath):
                os.remove(filepath)
            return

        if task_manager.check_timeout(task_id):
            task_manager.update_status(task_id, "timeout", 0, f"Timeout total ({TASK_MAX_DURATION_MINUTES}m)")
            if os.path.exists(filepath):
                os.remove(filepath)
            return

        task_manager.update_status(task_id, "uploading", 80, "Subiendo...")
        logger.info(f"[{task_id}] Iniciando subida a S3")

        try:
            timestamp = unique_filename.split('_')[0]
            output_filename = f"{timestamp}_{detected_object}.glb"
            s3_output_key = f"{S3_GENERATED_PREFIX}{output_filename}"
            logger.info(f"[{task_id}] Destino S3: {s3_output_key}")

            model_url = upload_model_to_s3(modified_model, s3_output_key, task_id=task_id)
            logger.info(f"[{task_id}] Modelo subido: {model_url}")
        except Exception as e:
            logger.error(f"[{task_id}] Excepción en subida: {str(e)}\n{traceback.format_exc()}")
            task_manager.update_status(task_id, "failed", 0, f"Error subiendo a S3: {str(e)[:100]}")
            if os.path.exists(filepath):
                os.remove(filepath)
            return

        logger.info(f"[{task_id}] Guardando en base de datos...")
        task_manager.update_status(task_id, "saving_db", 95, "Guardando en BD...")

        try:
            model_db_id = save_generated_model_to_db(
                task_id=task_id,
                detection_data=detection_data,
                model_url=model_url,
                model_key=s3_output_key,
                model_filename=output_filename,
                user_id=user_id  # ⭐ Pasar user_id
            )
            logger.info(f"[{task_id}] Modelo guardado en BD con ID: {model_db_id}")
        except Exception as e:
            logger.error(f"[{task_id}] Error guardando en BD (no crítico): {str(e)}")
            model_db_id = None  # No fallar todo el proceso si falla el guardado en BD

        # COMPLETADO
        if os.path.exists(filepath):
            os.remove(filepath)

        timings = task_manager.get_timings(task_id)
        total_time = sum(t.get("seconds", 0) for t in timings.values())

        logger.info(f"[{task_id}] Preparando respuesta final")
        task_manager.update_status(
            task_id,
            "completed",
            100,
            "Completado",
            detection=detection_data,
            model_url=model_url,
            model_key=s3_output_key,
            model_filename=output_filename,
            model_db_id=model_db_id,
            timings=timings
        )

        logger.info(f"[{task_id}] COMPLETADO ({total_time:.2f}s)")
        logger.info(f"[{task_id}] FIN")

    except TimeoutError as e:
        logger.error(f"[{task_id}] TIMEOUT: {str(e)}")
        task_manager.update_status(task_id, "timeout", 0, str(e))
        if os.path.exists(filepath):
            os.remove(filepath)

    except Exception as e:
        logger.error(f"[{task_id}] Error crítico: {str(e)}\n{traceback.format_exc()}")
        task_manager.update_status(task_id, "failed", 0, f"Error crítico: {str(e)[:100]}")
        if os.path.exists(filepath):
            os.remove(filepath)


TASK_STATES = {
    'PENDING': 'pending',
    'YOLO_DETECTING': 'yolo_detecting',
    'LOADING_MODEL': 'loading_model',
    'EXTRACTING_COLORS': 'extracting_colors',
    'MODIFYING_MODEL': 'modifying_model',
    'UPLOADING': 'uploading',
    'COMPLETED': 'completed',
    'FAILED': 'failed',
    'TIMEOUT': 'timeout',
    'CANCELLED': 'cancelled'
}


@model_bp.route('/process-image', methods=['POST'])
def process_image():
    """Endpoint para procesar imagen y generar modelo 3D (Asincrónico)"""
    if 'image' not in request.files:
        return jsonify({"message": "No se envió ninguna imagen"}), 400

    file = request.files['image']

    user_id = request.form.get('user_id', None)
    logger.info(f"Recibiendo imagen con user_id: {user_id}")

    if file.filename == '':
        return jsonify({"message": "Nombre de archivo vacío"}), 400

    if not allowed_file(file.filename):
        return jsonify({"message": "Formato de imagen no permitido"}), 400

    try:
        filename = secure_filename(file.filename)
        timestamp = str(int(np.random.random() * 1000000))
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(filepath)

        # Marcar como procesando
        processing_status[unique_filename] = {
            "status": "processing",
            "progress": 0,
            "current_step": "Guardando imagen...",
            "user_id": user_id
        }

        thread = Thread(target=process_image_background, args=(filepath, unique_filename, user_id))
        thread.daemon = True
        thread.start()

        return jsonify({
            "success": True,
            "message": "Procesamiento iniciado",
            "task_id": unique_filename,
            "status_url": f"/model/status/{unique_filename}"
        }), 202

    except Exception as e:
        return jsonify({
            "message": "Error al procesar la imagen",
            "error": str(e)
        }), 500


@model_bp.route('/status/<task_id>', methods=['GET'])
def get_processing_status(task_id):
    """Obtiene estado del procesamiento (API-safe para Android)"""

    if not task_manager.exists(task_id):
        return jsonify({
            "success": False,
            "status": "error",
            "message": "Tarea no encontrada"
        }), 404

    status = task_manager.get_status(task_id)

    PROCESSING_STATES = {
        "initialized",
        "yolo_detecting",
        "loading_model",
        "generating_model",
    }

    raw_state = status.get("state", "unknown")

    if raw_state in PROCESSING_STATES:
        public_state = "processing"
    elif raw_state == "completed":
        public_state = "completed"
    elif raw_state in {"failed", "error", "timeout", "cancelled"}:
        public_state = "error"
    else:
        public_state = "processing"  # fallback seguro

    start_time = status.get("start_time")
    elapsed = None
    if start_time:
        try:
            elapsed = (datetime.now() - datetime.fromisoformat(start_time)).total_seconds()
        except Exception:
            pass

    response = {
        "success": True,
        "status": public_state,
        "state": raw_state,
        "taskId": task_id,
        "progress": status.get("progress", 0),
        "currentStep": status.get("current_step") or "Procesando...",
        "elapsedSeconds": round(elapsed, 2) if elapsed else None,
        "timings": status.get("timings", {})
    }

    if public_state == "completed":
        response.update({
            "success": True,
            "detection": status.get("detection"),
            "modelUrl": status.get("model_url"),
            "model_url": status.get("model_url"),
            "modelKey": status.get("model_key"),
            "model_key": status.get("model_key"),
            "modelFilename": status.get("model_filename"),
            "model_filename": status.get("model_filename"),
            "modelDbId": status.get("model_db_id"),
            "model_db_id": status.get("model_db_id"),
            "timings": status.get("timings")
        })
        return jsonify(response), 200

    elif public_state == "error":
        response.update({
            "message": status.get("current_step", "Error durante el procesamiento")
        })

    return jsonify(response), 200


def save_generated_model_to_db(task_id, detection_data, model_url, model_key, model_filename, user_id=None):
    """
    Guarda el modelo generado en MongoDB

    Args:
        task_id: ID de la tarea
        detection_data: Datos de detección (objeto detectado, confianza, colores, etc.)
        model_url: URL del modelo en S3
        model_key: Clave del modelo en S3
        model_filename: Nombre del archivo del modelo
        user_id: ID del usuario que generó el modelo (opcional)

    Returns:
        String con ObjectId del documento insertado o None si falla
    """
    try:
        models_collection = mongo.db.models

        # Preparar datos del modelo
        model_data = {
            "name": f"{detection_data.get('object', 'model').title()} - Generado",
            "description": f"Modelo generado automáticamente detectando: {detection_data.get('object', 'unknown')} "
                           f"con confianza {detection_data.get('confidence', 0):.2%}",
            "category": detection_data.get('object', 'unknown').lower(),
            "imageUrl": "",  # TODO: Generar thumbnail del modelo
            "modelUrl": model_url,
            "modelKey": model_key,
            "modelFilename": model_filename,
            "rating": 0.0,
            "price": 0.0,
            "isActive": True,
            "userId": ObjectId(user_id) if user_id else None,
            "detectionData": {
                "object": detection_data.get('object'),
                "confidence": detection_data.get('confidence'),
                "bbox": detection_data.get('bbox'),
                "colors": detection_data.get('colors')
            },
            "taskId": task_id,
            "createdAt": datetime.utcnow(),
            "updatedAt": datetime.utcnow(),
            "status": "completed"
        }

        result = models_collection.insert_one(model_data)
        logger.info(f"[{task_id}] Modelo guardado en BD con ID: {result.inserted_id}")
        return str(result.inserted_id)

    except Exception as e:
        logger.error(f"[{task_id}] Error guardando modelo en BD: {str(e)}\n{traceback.format_exc()}")
        return None


def modify_3d_model(base_model_data, colors, scale_factor=1.0, task_id=None):
    """
    Modifica modelo 3D base aplicando colores y escala

    Args:
        base_model_data: BytesIO con el modelo base (no path!)
        colors: Lista de colores dominantes
        scale_factor: Factor de escala (1.0 = sin cambio)
        task_id: ID de la tarea (para logging)

    Returns:
        BytesIO con el modelo modificado
    """
    start = time.time()
    logger.info(f"[{task_id}] 🧱 Cargando modelo 3D con trimesh...")

    mesh, mesh_info = load_mesh_non_blocking(
        base_model_data,
        file_type='glb',
        task_id=task_id
    )

    # Validar que la carga fue exitosa
    if mesh is None:
        raise ValueError(f"[{task_id}] No se pudo cargar el modelo 3D")

    logger.info(f"[{task_id}] Modelo cargado en {time.time() - start:.2f}s")

    if scale_factor != 1.0:
        mesh.apply_scale(scale_factor)

    # Aplicar color dominante al modelo
    if colors and len(colors) > 0:
        try:
            # Convertir hex a RGB normalizado (0-1)
            primary_color = colors[0]["color"]
            r = int(primary_color[1:3], 16) / 255.0
            g = int(primary_color[3:5], 16) / 255.0
            b = int(primary_color[5:7], 16) / 255.0

            # Crear color RGBA
            color_rgba = np.array([r, g, b, 1.0])

            # Aplicar color a todos los vértices
            mesh.visual.vertex_colors = np.tile(color_rgba, (len(mesh.vertices), 1))
            logger.debug(f"[{task_id}] Color aplicado: {primary_color}")
        except Exception as e:
            logger.warning(f"[{task_id}] No se pudo aplicar color: {str(e)}")

    # Exportar a BytesIO
    output = BytesIO()
    mesh.export(output, file_type='glb')
    output.seek(0)

    logger.info(f"[{task_id}] Modelo exportado exitosamente")

    return output


def download_model_from_s3(object_key, task_id=None):
    """Descarga modelo 3D desde S3 con reintentos"""
    logger.info(f"[{task_id}] Intentando descargar: {object_key}")

    max_retries = 3
    retry_delay = 2  # segundos

    for attempt in range(max_retries):
        try:
            logger.info(f"[{task_id}] Intento {attempt + 1}/{max_retries}")

            response = s3_client.get_object(
                Bucket=S3_BUCKET_NAME,
                Key=object_key
            )

            logger.info(f"[{task_id}] Descarga completada")

            # Leer con chunks para evitar timeouts
            model_data = BytesIO()
            chunk_size = 1024 * 1024  # 1MB chunks

            for chunk in response['Body'].iter_chunks(chunk_size=chunk_size):
                model_data.write(chunk)

            model_data.seek(0)
            return model_data

        except (ClientError, ReadTimeoutError) as e:
            logger.warning(f"[{task_id}] Error en intento {attempt + 1}: {str(e)}")

            if attempt < max_retries - 1:
                import time
                logger.info(f"[{task_id}] Reintentando en {retry_delay}s...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Backoff exponencial
            else:
                logger.error(f"[{task_id}] Agotados todos los reintentos")
                return None

    return None


def upload_model_to_s3(file_data, object_key, task_id=None):
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
        logger.info(f"[{task_id}] Modelo subido a: {url}")
        return url
    except ClientError as e:
        raise Exception(f"Error subiendo a S3: {str(e)}")


def list_available_models_in_s3():
    """Listo modelo disponible en S3"""
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


@model_bp.route('/cache-stats', methods=['GET'])
def get_cache_stats():
    """
    Endpoint para obtener estadísticas del caché
    Útil para monitoreo
    """
    valid_detections = sum(1 for cache in detection_cache.values() if is_cache_valid(cache))

    return jsonify({
        "success": True,
        "cache_stats": {
            "total_detection_entries": len(detection_cache),
            "valid_detections": valid_detections,
            "expired_detections": len(detection_cache) - valid_detections,
            "model_cache_entries": len(model_cache),
            "cache_expiry_hours": CACHE_EXPIRY_HOURS,
            "max_cache_size": MAX_CACHE_SIZE
        }
    }), 200


@model_bp.route('/cache-clear', methods=['POST'])
def clear_caches():
    """
    Endpoint para limpiar cachés (útil para debugging)
    """
    global detection_cache, model_cache

    detection_cache.clear()
    model_cache.clear()

    return jsonify({
        "success": True,
        "message": "Cachés limpiados correctamente"
    }), 200


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


@model_bp.route('/catalog', methods=['GET'])
def get_catalog():
    """
    Endpoint para obtener el catálogo de productos/modelos
    """
    try:
        category = request.args.get('category', None)
        limit = int(request.args.get('limit', 20))
        offset = int(request.args.get('offset', 0))

        available_models = list_available_models_in_s3()

        catalog_items = []

        for i, model_name in enumerate(available_models[offset:offset + limit]):
            catalog_items.append({
                "id": f"model_{i + offset + 1}",
                "name": f"Modelo {model_name.title()}",
                "description": f"Descripción del modelo {model_name}",
                "imageUrl": f"https://via.placeholder.com/300x200/4A90E2/FFFFFF?text={model_name}",
                "rating": round(np.random.uniform(3.5, 5.0), 1),
                "price": round(np.random.uniform(50.0, 500.0), 2),
                "category": model_name.lower(),
                "isActive": True,
                "model3dKey": f"{S3_MODELS_PREFIX}{model_name}.glb"
            })

        return jsonify({
            "success": True,
            "models": catalog_items,
            "total": len(available_models),
            "limit": limit,
            "offset": offset
        }), 200

    except Exception as e:
        return jsonify({
            "message": "Error al obtener catálogo",
            "error": str(e)
        }), 500


@model_bp.route('/user/<user_id>/models', methods=['GET'])
def get_user_models(user_id):
    """Obtiene todos los modelos generados por un usuario"""
    try:
        from bson import ObjectId
        from datetime import datetime

        models_collection = mongo.db.models

        user_models = models_collection.find({
            'userId': ObjectId(user_id)
        }).sort('createdAt', -1)

        models_list = []
        for model in user_models:
            # ⭐ Convertir datetime a string
            created_at = model.get('createdAt')
            if isinstance(created_at, datetime):
                created_at = created_at.isoformat()
            elif isinstance(created_at, str):
                created_at = created_at
            else:
                created_at = None

            models_list.append({
                '_id': str(model['_id']),
                'name': model.get('name', ''),
                'description': model.get('description', ''),
                'category': model.get('category', ''),
                'imageUrl': model.get('imageUrl', ''),
                'modelUrl': model.get('modelUrl', ''),
                'modelKey': model.get('modelKey', ''),
                'modelFilename': model.get('modelFilename', ''),
                'price': model.get('price', 0.0),
                'rating': model.get('rating', 0.0),
                'isActive': model.get('isActive', True),
                'detectionData': model.get('detectionData', {}),
                'createdAt': created_at,  # ⭐ Ya es string
                'status': model.get('status', 'unknown')
            })

        return jsonify({
            'success': True,
            'models': models_list,
            'count': len(models_list)
        }), 200

    except Exception as e:
        logger.error(f"Error obteniendo modelos del usuario: {str(e)}")
        return jsonify({
            'error': 'Error al obtener modelos',
            'details': str(e)
        }), 500