# Build stage. Create the trained model once inside the image.
FROM python:3.11-slim AS build
WORKDIR /app
COPY data/ data/
COPY create_model.py .
RUN pip install --no-cache-dir pandas scikit-learn
RUN python create_model.py

# Runtime stage. Lightweight image that serves the API.
FROM python:3.11-slim
WORKDIR /app
COPY --from=build /app/model/ model/
COPY --from=build /app/data/ data/
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
ENV PORT=8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]