"""WSGI entry point for production deployments (gunicorn, uWSGI, etc.).

This module imports the Flask application object from app.py without touching
any model or inference code.  The model is loaded at module import time inside
app.py, which means gunicorn --preload triggers exactly one model load in the
master process before any worker forks.

Port is intentionally NOT configured here.  The caller (gunicorn CLI, a
Procfile, or a container CMD) supplies --bind 0.0.0.0:${PORT}.  This keeps the
binding config-driven (env var PORT → gunicorn --bind flag in the Dockerfile
CMD) and avoids hard-coding 7860 in application code.

Usage
-----
    gunicorn wsgi:application            # defaults from Dockerfile CMD
    PORT=8080 gunicorn --bind 0.0.0.0:${PORT} wsgi:application
"""

from app import app as application 
