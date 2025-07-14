import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd

class GrassRemover:
    def __init__(self, grass_colors=None, tolerance=25):
        if grass_colors is None:
            self.grass_colors = [
                (83, 108, 44), (100, 123, 57), (68, 93, 34), (92, 124, 52),
                (92, 116, 60), (60, 92, 28), (68, 84, 28), (82, 109, 37),
                (96, 121, 60), (91, 115, 52), (92, 124, 51), (100, 124, 52),
                (100, 132, 59), (76, 100, 20)
            ]
        else:
            self.grass_colors = grass_colors
        self.tolerance = tolerance
        self.additional_green_ranges = [
            ([45, 30, 40], [65, 255, 180]),   # Lighter grass tones
        ]

    def rgb_to_hsv_range(self, rgb_color):
        import colorsys
        r, g, b = rgb_color
        h, s, v = colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0)
        h_cv = int(h * 179)
        s_cv = int(s * 255)
        v_cv = int(v * 255)
        h_tol = min(15, self.tolerance // 2)
        s_tol = min(50, self.tolerance)
        v_tol = min(50, self.tolerance)
        lower = np.array([max(0, h_cv - h_tol), max(0, s_cv - s_tol), max(0, v_cv - v_tol)])
        upper = np.array([min(179, h_cv + h_tol), min(255, s_cv + s_tol), min(255, v_cv + v_tol)])
        return lower, upper

    def create_grass_mask(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for rgb in self.grass_colors:
            lower, upper = self.rgb_to_hsv_range(rgb)
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))
        for lower, upper in self.additional_green_ranges:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, np.array(lower), np.array(upper)))
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        return mask

class SaliencyColorClassifier:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Image at {image_path} could not be loaded.")
        self.mask = None
        self.dominant_color = None

    def process_single_image(self):
        remover = GrassRemover()
        self.mask = cv2.bitwise_not(remover.create_grass_mask(self.image))

    def extract_dominant_color(self, plot=False):
        image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        indices = np.column_stack(np.where(self.mask == 255))
        pixels_in_white = image[indices[:, 0], indices[:, 1]]
        if pixels_in_white.max() <= 1.0:
            pixels_in_white = (pixels_in_white * 255).astype(np.uint8)
        color_tuples = [tuple(pixel) for pixel in pixels_in_white]
        color_counter = Counter(color_tuples)
        most_common = color_counter.most_common(10)
        colors = [c[0] for c in most_common]
        counts = [c[1] for c in most_common]
       
        dominant_color = most_common[0][0]
        self.dominant_color = dominant_color
        return dominant_color

    def classify_color(self):
        color_rgb = np.uint8([[self.dominant_color]])
        color_hsv = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2HSV)[0][0]
        h, s, v = color_hsv
        if ((h >= 0 and h <= 10) or (h >= 170 and h <= 180)) and s > 70 and v > 50:
            return 'red'
        elif (h >= 180 and h <= 200) and s > 50 and v > 50:
            return 'sky_blue'
        elif s < 30 and v > 200:
            return 'white'
        elif v < 50:
            return 'black'
        else:
            reference_colors = {
                'red': np.array([0, 255, 255]),
                'green': np.array([60, 255, 255]),
                'white': np.array([0, 0, 255]),
                'black': np.array([0, 0, 0])
            }
            current_color = np.array([h, s, v])
            min_distance = float('inf')
            closest_color = 'other'
            for color_name, ref_hsv in reference_colors.items():
                if color_name == 'red':
                    hue_diff = min(abs(h - ref_hsv[0]), 180 - abs(h - ref_hsv[0]))
                    distance = np.sqrt(hue_diff**2 + (s - ref_hsv[1])**2 + (v - ref_hsv[2])**2)
                else:
                    distance = np.linalg.norm(current_color - ref_hsv)
                if distance < min_distance:
                    min_distance = distance
                    closest_color = color_name
            return closest_color

    def visualize(self):
        plt.figure(figsize=(12,3))
        plt.subplot(1,4,1)
        plt.imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")
        plt.axis('off')
        plt.subplot(1,4,2)
        plt.imshow(self.mask, cmap='jet')
        plt.title("Saliency Map")
        plt.axis('off')
        plt.subplot(1,4,3)
        masked_image = cv2.bitwise_and(self.image, self.image, mask=self.mask)
        plt.imshow(cv2.cvtColor(masked_image,cv2.COLOR_BGR2RGB), cmap='gray')
        plt.title("Salient Mask")
        plt.axis('off')
        plt.subplot(1,4,4)
        patch = np.zeros((100, 100, 3), dtype=np.uint8)
        patch[:] = self.dominant_color
        plt.imshow(patch)
        plt.title(f"Dominant Color\n({self.classify_color()})")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

class FolderColorAnalyzer:
    def __init__(self, folder_path, classifier_cls):
        self.folder_path = folder_path
        self.classifier_cls = classifier_cls
        self.rgb_values = []

    def analyze(self):
        self.rgb_values.clear()
        for img_name in sorted(os.listdir(self.folder_path)):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                img_path = os.path.join(self.folder_path, img_name)
                try:
                    scc = self.classifier_cls(img_path)
                    scc.process_single_image()
                    rgb = scc.extract_dominant_color()
                    
                    self.rgb_values.append(rgb)
                except Exception as e:
                    print(f"Error processing {img_name}: {e}")

    def get_rgb_statistics(self):
        if not self.rgb_values:
            return None, None, None
        arr = np.array(self.rgb_values)
        mean_rgb = tuple(np.mean(arr, axis=0).astype(int))
        median_rgb = tuple(np.median(arr, axis=0).astype(int))
        mode_rgb = Counter([tuple(x) for x in arr]).most_common(1)[0][0]
        return mean_rgb, mode_rgb, median_rgb

class TrackedObjectsAnalyzer:
    def __init__(self, root_folder, classifier_cls):
        self.root_folder = root_folder
        self.classifier_cls = classifier_cls
        self.folder_colors = []

    def analyze_all(self):
        self.folder_colors.clear()
        for folder_name in sorted(os.listdir(self.root_folder)):
            folder_path = os.path.join(self.root_folder, folder_name)
            if os.path.isdir(folder_path):
                analyzer = FolderColorAnalyzer(folder_path, self.classifier_cls)
                analyzer.analyze()
                mean_rgb, mode_rgb, median_rgb = analyzer.get_rgb_statistics()
                self.folder_colors.append((folder_name, mean_rgb, mode_rgb, median_rgb))

    def print_summary_table(self):
        print(f"{'Folder Name':<30} {'Mean RGB':<20} {'Mode RGB':<20} {'Median RGB':<20}")
        print('-' * 90)
        for folder_name, mean_rgb, mode_rgb, median_rgb in self.folder_colors:
            print(f"{folder_name:<30} {str(mean_rgb):<20} {str(mode_rgb):<20} {str(median_rgb):<20}")

    def save_csv(self, csv_path="dominant_colors_stats.csv"):
        import csv
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["id_name", "mean_rgb", "mode_rgb", "median_rgb"])
            for folder_name, mean_rgb, mode_rgb, median_rgb in self.folder_colors:
                writer.writerow([folder_name, str(mean_rgb), str(mode_rgb), str(median_rgb)])
        print(f"CSV saved to {csv_path}")
    def to_dataframe(self):
        columns = ["id_name", "mean_rgb", "mode_rgb", "median_rgb"]
        return pd.DataFrame(self.folder_colors, columns=columns)    

# if __name__ == "__main__":
#     # ... code ...
#     root_folder_1 = "/home/levi/Documents/pipecat/pipecat_quick_Start/broadcast_object-20250713T171653Z-1-001/broadcast_object"
#     analyzer_1 = TrackedObjectsAnalyzer(root_folder_1, SaliencyColorClassifier)
#     analyzer_1.analyze_all()


#     analyzer_1.save_csv("dominant_1.csv")
#     df1=analyzer_1.to_dataframe()
   
#     df1['source'] = 'first'  # Add source column
#     print(df1)

    

