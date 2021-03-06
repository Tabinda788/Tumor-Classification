from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = tf.keras.models.load_model(f"saved_model_new")

CLASS_NAMES = ["no", "yes"]

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    #print("PRINTING THE SIZE OF IMAGE",image.shape)
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):  
    try:
        image = read_file_as_image(await file.read())
        img_batch = np.expand_dims(image, 0)
        predictions = MODEL.predict(img_batch)
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])
        if confidence>0.5:
            predicted_class = "The image has a tumor"
        else:
            predicted_class = "The image has no tumor"

        return {
            'class': predicted_class,
            'confidence': float(confidence)
        }

    except:
        return {
            'Message': "Please try with some other image!!",
        }


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)