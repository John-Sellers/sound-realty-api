import modal

app = modal.App("sound-realty-api")

# Use your local Dockerfile to build the image on Modal's side.
image = modal.Image.from_dockerfile("Dockerfile")


@app.function(image=image)
@modal.asgi_app()
def fastapi_app():
    # The Dockerfile already copied app.py and built the model.
    import app as myapp

    return myapp.app
