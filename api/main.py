
from fastapi import FastAPI, UploadFile, File
import uvicorn
import numpy
from io import BytesIO
import tensorflow as tf
from PIL import Image

app = FastAPI()


model = tf.keras.models.load_model("../Models/1")
classes = ['Tomato_Bacterial_spot',
 'Tomato_Early_blight',
 'Tomato_Late_blight',
 'Tomato_Leaf_Mold',
 'Tomato_Septoria_leaf_spot',
 'Tomato_Spider_mites_Two_spotted_spider_mite',
 'Tomato__Target_Spot',
 'Tomato__Tomato_YellowLeaf__Curl_Virus',
 'Tomato__Tomato_mosaic_virus',
 'Tomato_healthy']
@app.get("/ping")
async def ping():
    return "Test"

def read_image(data)-> numpy.ndarray:       #data is bytes as input
    image =numpy.array(Image.open(BytesIO(data))
@app.post("/predict")
async def predict(
        file: UploadFile = File(...)
):
    image = read_image(file.read())

    img_batch = numpy.expand_dims()             #model.predict only accepts image batches, current image is[256,256,3] so converting to higher dimensional array [[256,256,3]]

    prediction = model.predict(img_batch)


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)