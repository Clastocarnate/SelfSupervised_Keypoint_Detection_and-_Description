import cv2
import os

def process_images(folder_path):
    sift = cv2.SIFT_create()
    orb = cv2.ORB_create()

    sift_keypoints_total = 0
    orb_keypoints_total = 0
    image_count = 0

    with open("sift_keypoints.txt", "w") as sift_file, open("orb_keypoints.txt", "w") as orb_file:
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(folder_path, filename)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                if image is None:
                    continue
                
                # Detect SIFT keypoints
                sift_keypoints = sift.detect(image, None)
                sift_file.write("{}: {} keypoints\n".format(filename, len(sift_keypoints)))

                # Detect ORB keypoints
                orb_keypoints = orb.detect(image, None)
                orb_file.write("{}: {} keypoints\n".format(filename, len(orb_keypoints)))

                sift_keypoints_total += len(sift_keypoints)
                orb_keypoints_total += len(orb_keypoints)
                image_count += 1

    if image_count > 0:
        print("SIFT - Total keypoints: {}, Average per image: {}".format(sift_keypoints_total, sift_keypoints_total / image_count))
        print("ORB - Total keypoints: {}, Average per image: {}".format(orb_keypoints_total, orb_keypoints_total / image_count))
    else:
        print("No images processed.")

# Example usage
folder_path = 'images'
process_images(folder_path)
