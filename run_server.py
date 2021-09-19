from fastapi import FastAPI, Body
import cv2
from apply_mask import apply_mask
from starlette.responses import StreamingResponse

import io
import numpy as np
import base64

app = FastAPI()


@app.post("/apply_mask")
async def create_upload_file(image: str = Body(...), mask_name: str = "shrek"):
    print("mask_name", mask_name)
    encoded_data = image.split(',')[1]
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # image = cv2.imread(file)
    processed = apply_mask(image, mask_name)
    res, im_png = cv2.imencode(".png", processed)

    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")
