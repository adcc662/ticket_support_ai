FROM python:3.13-slim

# Configurar variables de entorno
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    DEBIAN_FRONTEND=noninteractive \
    ACCEPT_EULA=Y


COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Crear directorio de trabajo
WORKDIR /app


COPY pyproject.toml uv.lock* ./


RUN uv sync --frozen --no-install-project


COPY . .

# Exponer puerto
EXPOSE 8000

# Comando por defecto para desarrollo con hot-reload
CMD ["uv", "run", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]