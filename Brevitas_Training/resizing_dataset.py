import os
from PIL import Image

def resize_images(input_folder, output_folder, size=(32, 32)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith((".jpg", ".jpeg", ".png", ".bmp", ".ppm")):
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path)
            img_resized = img.resize(size, Image.LANCZOS)

            # Change the extension to .jpg
            base_name = os.path.splitext(filename)[0]
            output_path = os.path.join(output_folder, base_name + ".jpg")
            img_resized.save(output_path, "JPEG")
            print(f"Saved resized image to {output_path}")

input_folder = "/home/ba/PycharmProjects/Brevitas_Thesis/Brevitas_Training/data/gtsrb/GTSRB/Final_Test/Images"
output_folder = "/home/ba/PycharmProjects/Brevitas_Thesis/Brevitas_Training/data/gtsrb/GTSRB_Resized"

resize_images(input_folder, output_folder)
print("Resizing is done.")