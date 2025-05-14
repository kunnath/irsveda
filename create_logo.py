#!/bin/bash

# Create a basic iris logo
python -c '
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os

# Create a 300x300 image with a white background
width, height = 300, 300
image = Image.new("RGB", (width, height), "white")
draw = ImageDraw.Draw(image)

# Draw an iris
center_x, center_y = width // 2, height // 2
outer_radius = 120
pupil_radius = 40

# Draw iris circle (blue-green gradient)
for r in range(pupil_radius, outer_radius):
    # Create a nice blue-green gradient
    ratio = (r - pupil_radius) / (outer_radius - pupil_radius)
    blue = int(200 * (1 - ratio))
    green = int(180 * ratio)
    color = (0, green, blue)
    
    # Draw circle
    draw.ellipse(
        (center_x - r, center_y - r, center_x + r, center_y + r),
        outline=color
    )

# Draw pupil (black)
draw.ellipse(
    (center_x - pupil_radius, center_y - pupil_radius, 
     center_x + pupil_radius, center_y + pupil_radius),
    fill="black"
)

# Add text
text = "AyushIris"
# Try to use a font if available, otherwise skip text
try:
    if os.path.exists("/System/Library/Fonts/Helvetica.ttc"):
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 40)
        draw.text((width//2, height-40), text, fill="darkgreen", font=font, anchor="ms")
    else:
        draw.text((width//2, height-40), text, fill="darkgreen", anchor="ms")
except:
    pass

# Save the image
image.save("static/iris_logo.png")
'
