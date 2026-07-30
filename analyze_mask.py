import cv2
import glob

# Load a decision frame
files = glob.glob('C:/Users/Hema Pandey/.gemini/antigravity-cli/brain/81819823-12da-4acf-95bd-49997ea1f6c8/Camera_+_Decisions_*.jpg')
if files:
    img = cv2.imread(files[0])
    # The error is written on the image by cv2.putText near the bottom.
    # But since it's hard to OCR, let's just analyze the Black_Mask image instead.
    print(f"Found {len(files)} decision frames.")
else:
    print("No decision frames found.")

mask_files = glob.glob('C:/Users/Hema Pandey/PycharmProjects/TemuFollower/artifacts/Black_Mask_*.jpg')
if mask_files:
    mask = cv2.imread(mask_files[0], cv2.IMREAD_GRAYSCALE)
    M = cv2.moments(mask)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        # ROI config:
        x0 = int(640 * 0.05)
        x1 = int(640 * 0.95)
        roi_w = x1 - x0
        center = x0 + roi_w / 2
        
        # Original logic:
        # line_x_in_roi = cX - x0
        # error = (line_x_in_roi - roi_w/2) / (roi_w/2)
        
        err = (cX - center) / (roi_w / 2)
        
        print(f"Analyzed {mask_files[0]}:")
        print(f"Centroid X: {cX}")
        print(f"ROI center: {center}")
        print(f"Computed error: {err:.4f}")
    else:
        print("Empty mask.")
else:
    print("No mask files found.")
