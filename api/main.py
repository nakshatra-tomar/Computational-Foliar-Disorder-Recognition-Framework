
from fastapi import FastAPI, UploadFile, File
import uvicorn
import numpy
from io import BytesIO
from PIL import Image

app = FastAPI()


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


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)