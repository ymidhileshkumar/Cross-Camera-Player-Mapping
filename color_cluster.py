import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import ast
from color_analysis import TrackedObjectsAnalyzer, SaliencyColorClassifier

class ColorClusterAnalyzer:
    def __init__(self, root_folder_1, root_folder_2, n_clusters=4):
        self.root_folder_1 = root_folder_1
        self.root_folder_2 = root_folder_2
        self.n_clusters = n_clusters
        self.combined_df = pd.DataFrame()

    def _analyze_folder(self, root_folder, source_label):
        analyzer = TrackedObjectsAnalyzer(root_folder, SaliencyColorClassifier)
        analyzer.analyze_all()
        df = analyzer.to_dataframe()
        df['source'] = source_label
        df.rename(columns={'median_rgb': 'Median RGB'}, inplace=True)
        return df

    def _parse_rgb(self, rgb_value):
        try:
            if isinstance(rgb_value, str):
                cleaned = rgb_value.replace("np.int64", "").replace("(", "").replace(")", "")
                return tuple(map(int, ast.literal_eval(cleaned)))
            return tuple(rgb_value)
        except:
            return (0, 0, 0)

    def run(self):
        df1 = self._analyze_folder(self.root_folder_1, 'first')
        df2 = self._analyze_folder(self.root_folder_2, 'second')
        self.combined_df = pd.concat([df1, df2], ignore_index=True)

        self.combined_df['Median RGB'] = self.combined_df['Median RGB'].apply(self._parse_rgb)
        median_colors = np.array(self.combined_df['Median RGB'].tolist())

        kmeans = KMeans(n_clusters=self.n_clusters, random_state=0, n_init=10)
        self.combined_df['cluster'] = kmeans.fit_predict(median_colors)
        self.combined_df['display_id'] = self.combined_df['source'] + '_' + self.combined_df['id_name'].astype(str)

    def save_and_return_results(self, filename='final_new.csv'):
        if self.combined_df.empty:
            raise ValueError("Run `run()` before saving.")
        self.combined_df[['id_name', 'source', 'cluster', 'Median RGB']].to_csv(filename, index=False)
        print(f"Saved cluster assignments to {filename}")
        return self.combined_df    

    def plot_clusters(self, output_file='color_clusters_1.png'):
        if self.combined_df.empty:
            raise ValueError("Run `run()` before plotting.")
        fig, axs = plt.subplots(nrows=self.n_clusters, figsize=(12, self.n_clusters * 2))
        for cluster_id in range(self.n_clusters):
            cluster_df = self.combined_df[self.combined_df['cluster'] == cluster_id]
            cluster_colors = cluster_df['Median RGB'].tolist()
            display_ids = cluster_df['display_id'].tolist()
            n_colors = len(cluster_colors)

            img = np.zeros((20, n_colors * 40, 3), dtype=np.uint8)
            for i, color in enumerate(cluster_colors):
                img[:, i*40:(i+1)*40, :] = color

            axs[cluster_id].imshow(img)
            axs[cluster_id].axis('off')
            axs[cluster_id].set_title(f"Cluster {cluster_id}")

            for i, did in enumerate(display_ids):
                axs[cluster_id].text(
                    i*40 + 20,
                    30,
                    did,
                    fontsize=8,
                    ha='center',
                    va='top',
                    rotation=90
                )

        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        print(f"Saved color clusters to {output_file}")
