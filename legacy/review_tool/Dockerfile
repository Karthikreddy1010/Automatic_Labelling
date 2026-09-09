FROM python:3.11-slim
WORKDIR /app
# stdlib-only app: just copy everything (images baked into the image)
COPY app.py candidates.json ./
COPY images ./images
ENV PORT=8080 REVIEW_DATA_DIR=/data
EXPOSE 8080
CMD ["python", "app.py"]
