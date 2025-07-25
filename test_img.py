import cv2
import sys
import numpy as np

# Replace these with your actual file paths
image_path1 = "/home/justin/code/BundleSDF/data/2022-11-18-15-10-24_milk/depth/1668813025164826994.png"
image_path2 = "/home/justin/code/Manipulator-Software/data/object_mesh/mouse_box_data/depth/000000.png"

# Load the images
img1 = cv2.imread(image_path1, cv2.IMREAD_UNCHANGED)
img2 = cv2.imread(image_path2, cv2.IMREAD_UNCHANGED)


def get_image_format(path):
    ext = path.split(".")[-1].lower()
    return ext


print(f"Image 1 path: {image_path1}")
print(f"Image 1 format: {get_image_format(image_path1)}")
if img1 is None:
    print("Failed to load image 1.")
else:
    print(f"Image 1 shape: {img1.shape}, dtype: {img1.dtype}")

print(f"\nImage 2 path: {image_path2}")
print(f"Image 2 format: {get_image_format(image_path2)}")
if img2 is None:
    print("Failed to load image 2.")
else:
    print(f"Image 2 shape: {img2.shape}, dtype: {img2.dtype}")

# Check if both images have the same shape and dtype
if img1 is not None and img2 is not None:
    same_shape = img1.shape == img2.shape
    same_dtype = img1.dtype == img2.dtype
    print(f"\nSame shape: {same_shape}")
    print(f"Same dtype: {same_dtype}")
else:
    print("\nCannot compare shape and dtype because one or both images failed to load.")


def print_depth_info(img, label):
    if img is not None:
        min_val = np.min(img)
        max_val = np.max(img)
        sample_vals = img.flatten()[:: max(1, img.size // 10)][:10]  # up to 10 samples
        print(f"{label} min: {min_val}, max: {max_val}")
        print(f"{label} sample values: {sample_vals}")
        if img.dtype in [np.uint16, np.int16, np.uint32, np.int32]:
            if max_val > 100:  # heuristic threshold
                print(f"{label} likely in millimeters (mm)")
            else:
                print(f"{label} may be in meters or another unit")
        elif img.dtype in [np.float32, np.float64]:
            if max_val < 10:
                print(f"{label} likely in meters")
            else:
                print(f"{label} may be in millimeters (mm) or another unit")
    else:
        print(f"{label} not loaded.")


print_depth_info(img1, "Image 1")
print_depth_info(img2, "Image 2")
