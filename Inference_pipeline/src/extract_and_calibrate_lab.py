from segment_anything import sam_model_registry, SamPredictor
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from skimage import io, color
from skimage.io import imsave
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
import shutil
import joblib
import torch
import cv2
import os
import warnings
warnings.filterwarnings("ignore", message=".*is a low contrast image*")


def extract_and_calibrate_lab():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # sam = sam_model_registry["vit_b"](checkpoint="./models/sam_vit_b_01ec64.pth")
    # sam = sam_model_registry["vit_l"](checkpoint="./models/sam_vit_l_0b3195.pth")
    sam = sam_model_registry["vit_h"](checkpoint="./models/sam_vit_h_4b8939.pth")
    sam.to(device)
    predictor = SamPredictor(sam)

    checker_path = next(Path("./input").glob("color_checker.*"), None)
    patches_dir = r"./output/patches"
    os.makedirs(patches_dir, exist_ok=True)

    img = cv2.imread(checker_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_h, img_w = img_rgb.shape[:2]
    cell_h, cell_w = img_h // 4, img_w // 6

    predictor.set_image(img_rgb)
    patch_id = 1
    for i in tqdm(range(24), desc="⏳ Extracting patches"):
            r = i // 6
            c = i % 6
            x1 = c * cell_w
            y1 = r * cell_h
            x2 = (c + 1) * cell_w
            y2 = (r + 1) * cell_h

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            input_point = np.array([[cx, cy]])
            input_label = np.array([1]) 

            masks, scores, logits = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True
            )

            best_mask = masks[np.argmax(scores)].astype(np.uint8)

            ys, xs = np.where(best_mask)          

            if ys.size == 0:
                raise ValueError("One of the patch mask is empty!")

            y1, y2 = ys.min(), ys.max() + 1
            x1, x2 = xs.min(), xs.max() + 1
            cy = int((y1 + y2) / 2)   
            cx = int((x1 + x2) / 2)
            y1, y2 = cy-25, cy+25
            x1, x2 = cx-25, cx+25

            patch_name = f"patch_{patch_id:02d}.png"
            imsave(os.path.join(patches_dir, patch_name), img_rgb[y1:y2, x1:x2])
            patch_id += 1

    print(f"🧩 Extracted and Saved {patch_id - 1} Patches to: {patches_dir}")


    lab_patch_path = r'./output/patches/lab_patch.csv'

    lab_summary = []

    for filename in os.listdir(patches_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            try:
                image_path = os.path.join(patches_dir, filename)
                image = io.imread(image_path)

                lab_image = color.rgb2lab(image)

                L_values = lab_image[:, :, 0].flatten()
                a_values = lab_image[:, :, 1].flatten()
                b_values = lab_image[:, :, 2].flatten()

                stats = {
                    'Image': filename,
                    'L': np.mean(L_values),
                    'A': np.mean(a_values),
                    'B': np.mean(b_values),
                }

                lab_summary.append(stats)

            except Exception as e:
                print(f"Error processing {filename}: {e}")

    df_summary = pd.DataFrame(lab_summary)
    df_summary = df_summary.sort_values(by='Image')
    df_summary.to_csv(lab_patch_path, index=False)


    y = pd.read_csv(r"./assets/ref_checker_LAB.csv")
    y = y[['L', 'A', 'B']]
    X =  pd.read_csv(r"./output/patches/lab_patch.csv")
    X = X[['L', 'A', 'B']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    n_components = 3
    pls_model = PLSRegression(n_components=n_components)
    pls_model.fit(X_train, y_train)

    valid_exts = ('.jpg', '.jpeg', '.png')
    img_path = [f"./output/images/{file}" for file in os.listdir("./output/images") if file.lower().endswith(valid_exts)]
    seg_path = [path.replace("/output/images", "/output/segmentation") for path in img_path]
    filenames = [path.split('/')[-1].split('.')[0] for path in img_path]

    names = []
    L_initial = []
    A_initial = []
    B_initial = []
    L_calib = []
    A_calib = []
    B_calib = []
    for i in tqdm(range(len(filenames)), desc="⏳ Extracting and Calibrating LAB"):
        filename = filenames[i]
        p1 = img_path[i]
        p2 = seg_path[i]
        rgb_image = io.imread(p1)
        mask_image = io.imread(p2)

        if rgb_image.shape != mask_image.shape:
            print("❌ Error: Shape mismatch between rgb_image and mask_image. Please verify the images and rerun the pipeline.")

        if mask_image.dtype == np.float64:
            mask_image = (mask_image * 255).astype(np.uint8)

        lower = np.array([0, 245, 0])
        upper = np.array([10, 255, 10])
        green_mask = (
            (mask_image[:, :, 0] >= lower[0]) & (mask_image[:, :, 0] <= upper[0]) &
            (mask_image[:, :, 1] >= lower[1]) & (mask_image[:, :, 1] <= upper[1]) &
            (mask_image[:, :, 2] >= lower[2]) & (mask_image[:, :, 2] <= upper[2])
        )
        green_mask_visual = green_mask.astype(np.uint8) * 255

        if not np.any(green_mask):
            print(f"❌ No segmentation mask found in {p2}, skipping.")
            continue

        lab_image = color.rgb2lab(rgb_image)

        L_masked = lab_image[:, :, 0][green_mask]
        A_masked = lab_image[:, :, 1][green_mask]
        B_masked = lab_image[:, :, 2][green_mask]

        lab_masked = np.stack([L_masked, A_masked, B_masked], axis=1)

        names.append(filename)
        L_initial.append(np.mean(lab_masked[:, 0]))
        A_initial.append(np.mean(lab_masked[:, 1]))
        B_initial.append(np.mean(lab_masked[:, 2]))
        lab_masked = pd.DataFrame(lab_masked, columns=X_train.columns)
        lab_calib = pls_model.predict(lab_masked)
        L_calib.append(np.mean(lab_calib[:, 0]))
        A_calib.append(np.mean(lab_calib[:, 1]))
        B_calib.append(np.mean(lab_calib[:, 2]))

    df = {'Name': names, 'L_initial': L_initial, 'A_initial': A_initial, 'B_initial': B_initial,
          'L_calib': L_calib, 'A_calib': A_calib, 'B_calib': B_calib}
    df = pd.DataFrame(df)
    df.to_csv('./output/results.csv', index=False)

    # y_pred = pls_model.predict(X_test)
    # r_squared = pls_model.score(X_test, y_test)
    # print(f"R-Squared: {r_squared}")
    # mse = mean_squared_error(y_test, y_pred)
    # print(f"Mean Squared Error: {mse}")