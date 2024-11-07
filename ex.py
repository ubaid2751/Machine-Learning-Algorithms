from PIL import Image
import numpy as np
from imageCompression import load_and_resize_image, compress_image
import matplotlib.pyplot as plt

# Load the image using PIL and convert to a NumPy array
image = Image.open('E:\Image Compression\Photo.jpg')
np_img = np.array(image)
print(np_img.shape)

# Load and resize the image using your functions
resized_img = load_and_resize_image(np_img)
compressed_img = compress_image(np_img, 16)

# Visualize the compressed image using matplotlib
plt.imshow(compressed_img)
plt.title("Compressed Image")
plt.axis("off")
plt.show()

output_path = 'E:\Image Compression\Compressed_Photo.jpg'
plt.imsave(output_path, compressed_img)

print(f"Compressed image saved to {output_path}")
