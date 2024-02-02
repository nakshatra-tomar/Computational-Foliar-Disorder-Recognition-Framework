
from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy
from io import BytesIO
import tensorflow as tf
from PIL import Image

app = FastAPI()


model = tf.keras.models.load_model("../Models/1")
classes = ['Tomato Bacterial Spot',
 'Tomato Early blight',
 'Tomato Late blight',
 'Tomato Leaf Mold',
 'Tomato Septoria leaf spot',
 'Tomato Spider mites Two spotted spider mite',
 'Tomato Target Spot',
 'Tomato YellowLeaf Curl Virus',
 'Tomato Tomato mosaic virus',
 'Tomato healthy']
@app.get("/ping")
async def ping():
    return "Test"

def read_image(data)-> numpy.ndarray:       #data is bytes as input
    image =numpy.array(Image.open(BytesIO(data)))
    return image
@app.post("/predict")
async def predict(
        file: UploadFile = File(...)
):
    image = read_image(await file.read())

    img_batch = numpy.expand_dims(image, 0)             #model.predict only accepts image batches, current image is[256,256,3] so converting to higher dimensional array [[256,256,3]]

    prediction = model.predict(img_batch)

    index = numpy.argmax(prediction[0])

    predicted_class = classes[index]
    confidence = numpy.max(prediction[0])

    return{
        'class' : predicted_class,
        'confidence' : float(confidence)
    }


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)