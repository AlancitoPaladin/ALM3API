# gunicorn.conf.py
import multiprocessing
import os

# Configuración de workers
workers = 1  # Solo 1 worker para evitar OOM en Render free tier
worker_class = 'sync'
worker_connections = 1000

# CRÍTICO: Aumentar timeout para procesamiento de imágenes
timeout = 300  # 5 minutos (era 30 segundos por defecto)
graceful_timeout = 300
keepalive = 5

# Límites de memoria y requests
max_requests = 100  # Reiniciar worker después de N requests
max_requests_jitter = 10

# Logging
accesslog = '-'
errorlog = '-'
loglevel = 'info'

# Configuración de bind (Render usa PORT env var)
bind = f"0.0.0.0:{os.getenv('PORT', '10000')}"

# Preload app para ahorrar memoria
preload_app = False  # False para evitar problemas con threads

print(f"🚀 Gunicorn configurado: timeout={timeout}s, workers={workers}")