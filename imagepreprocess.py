
import cv2
import numpy as np

def preprocess_image_for_ctpn(image_path, output_path=None):
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur to reduce noise and improve edge detection
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Adaptive Thresholding with adjusted parameters
    thresh_image = cv2.adaptiveThreshold(
        blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 4  # Adjust these parameters
    )

    # Apply Morphological operations to reduce noise
    kernel = np.ones((2, 2), np.uint8)
    morph_image = cv2.morphologyEx(thresh_image, cv2.MORPH_CLOSE, kernel)

    # Resizing (height normalization for CTPN, typically around 720p or 1080p)
    target_height = 720  # Adjust depending on your CTPN model's input expectations
    height, width = morph_image.shape
    scaling_factor = target_height / height
    target_width = int(width * scaling_factor)
    resized_image = cv2.resize(morph_image, (target_width, target_height), interpolation=cv2.INTER_AREA)

    # Save the preprocessed image if an output path is provided
    if output_path:
        cv2.imwrite(output_path, resized_image)

    return resized_image

# Example Usage
input_image_path = r"C:\Users\DURGA PRASAD\testimage2.jpg"
output_image_path = r"C:\Users\DURGA PRASAD\preprocessed_image_for_ctpnfinal.png"

preprocessed_image = preprocess_image_for_ctpn(input_image_path, output_image_path)

# Display the preprocessed image for visual verification (Optional)
cv2.imshow('Preprocessed Image for CTPN', preprocessed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()