# Tiny image: the control service is stdlib-only. `tzdata` is the sole dep (for ZoneInfo timezones).
FROM python:3.12-slim
RUN pip install --no-cache-dir tzdata
WORKDIR /app
COPY control.py postiz_client.py ./
EXPOSE 8770
CMD ["python", "control.py"]
