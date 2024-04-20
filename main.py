import cv2
import sys
import numpy as np
import requests
from PIL import Image
from io import BytesIO
from fastapi import FastAPI, HTTPException
import os

model_file = "opencv_face_detector_uint8.pb"
config_file = "opencv_face_detector.pbtxt"
net = cv2.dnn.readNetFromTensorflow(model_file, config_file)

app = FastAPI()

def blur_faces(image_url):
    # Load the image from the URL
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))

    # Check if the image is not JPEG
    if img.format != "JPEG":
        # Convert the image to JPEG
        with BytesIO() as f:
            img.save(f, format="JPEG")
            img = Image.open(f)

    # Convert the image to OpenCV format
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    if img is None:
        raise HTTPException(status_code=400, detail="Unable to load image file.")

    # Create a blob from the image
    blob = cv2.dnn.blobFromImage(
        cv2.resize(img, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0),
        False,
        False,
    )

    # Set the blob as input to the network
    net.setInput(blob)

    # Perform a forward pass of the network to get the detections
    detections = net.forward()

    # Iterate over the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # If the confidence of the detection is above 0.5, blur the detected face
        if confidence > 0.5:
            # Get the coordinates of the bounding box around the face
            box = detections[0, 0, i, 3:7] * np.array(
                [img.shape[1], img.shape[0], img.shape[1], img.shape[0]]
            )
            (startX, startY, endX, endY) = box.astype("int")

            # Blur the face in the image
            face = img[startY:endY, startX:endX]
            blurred_face = cv2.GaussianBlur(face, (99, 99), 30)
            img[startY:endY, startX:endX] = blurred_face

    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # Save the processed image to a file
    output_filename = "output.jpg"
    img.save(output_filename)
    return output_filename


import requests

def upload_to_server(image_path, upload_url):
    with open(image_path, "rb") as f:
        files = {"photo": f}
        response = requests.post(upload_url, files=files)
        return response.json()


@app.post("/blur")
async def blur_face_endpoint(image_data: dict):
    if "url" not in image_data:
        raise HTTPException(status_code=400, detail="Image URL is missing in request body.")

    image_url = image_data["url"]
    output_file = blur_faces(image_url)
    upload_url = "http://ec2-43-204-100-197.ap-south-1.compute.amazonaws.com:3000/upload"
    upload_response = upload_to_server(output_file, upload_url)

    # Clean up the temporary output file
    os.remove(output_file)

    return {"upload_response": upload_response}
