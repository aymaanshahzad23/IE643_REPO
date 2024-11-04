from PIL import Image

# Load the image
image = Image.open('nice.png')

# Convert to grayscale
bw_image = image.convert('L')

# Define a threshold value
threshold = 130  # You can adjust this value

# Apply the threshold
binary_image = bw_image.point(lambda p: 255 if p > threshold else 0)

# Save the binary image
binary_image.save('nice_new.png')
