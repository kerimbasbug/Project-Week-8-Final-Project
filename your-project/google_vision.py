import io
import os
import time
# Imports the Google Cloud client library
from google.cloud import vision
from google.cloud.vision import types

#GOOGLE_APPLICATION_CREDENTIALS = "/Users/kerimbasbug/PycharmProjects/IronHack_Lessons/finalproject-68cadd08a17c.json"

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/Users/kerimbasbug/PycharmProjects/IronHack_Lessons/finalproject-68cadd08a17c.json"

# Instantiates a client
client = vision.ImageAnnotatorClient()

start = time.time()
# The name of the image file to annotate
file_name = os.path.join(
    os.path.dirname(__file__),
    'taco2.jpg')

# Loads the image into memory
with io.open(file_name, 'rb') as image_file:
    content = image_file.read()

image = types.Image(content=content)

# Performs label detection on the image file
response = client.label_detection(image=image)
labels = response.label_annotations

print('Labels:')
for label in labels:
    print(label.description)

end = time.time()
print(end - start)