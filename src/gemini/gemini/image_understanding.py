from google import genai
from google.genai import types
import os
import json

GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
client = genai.Client(api_key=GOOGLE_API_KEY)

with open('/home/trace/robot/fridge.jpg', 'rb') as f:
    image_bytes = f.read()

response = client.models.generate_content(
model='gemini-2.0-flash',
contents=[
    types.Part.from_bytes(
    data=image_bytes,
    mime_type='image/jpeg',
    ),
    'Return the bounding box of the sprite.'
]
)

print(response.text)

json_output = response.text

lines = json_output.splitlines()
for i, line in enumerate(lines):
    if line == "```json":
        json_output = "\n".join(lines[i+1:])  # Remove everything before "```json"
        json_output = json_output.split("```")[0]  # Remove everything after the closing "```"
        break  # Exit the loop once "```json" is found

print(json_output)

bounding_boxes = json_output

for i, bounding_box in enumerate(json.loads(bounding_boxes)):

      # Convert normalized coordinates to absolute coordinates
      abs_y1 = int(bounding_box["box_2d"][0])
      abs_x1 = int(bounding_box["box_2d"][1])
      abs_y2 = int(bounding_box["box_2d"][2])
      abs_x2 = int(bounding_box["box_2d"][3])

print(f"ymin: {abs_y1}, xmin: {abs_x1}, ymax: {abs_y2}, xmax: {abs_x2}")

center = ((abs_y1 + abs_y2) // 2, (abs_x1 + abs_x2) // 2)
print(f"Center of bounding box: {center}")

