
import os
import csv
import time
import cv2
import numpy as np
import pandas as pd
from glob import glob
from typing import List, Tuple, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.cluster import KMeans
from tqdm import tqdm

class ImageFolderAnalyzer:
    """Analyze image folders for image counts and clustering statistics."""
    
    def __init__(self,csv_path: str, parent_dir: str, indi: str):
        self._csv_path =csv_path
        self._parent_dir = parent_dir
        self._filtered_df: Optional[pd.DataFrame] = None
        self._results_df: Optional[pd.DataFrame] = None
        self._low_count_stats: Optional[pd.DataFrame] = None
        self._low_count_kmeans: Optional[pd.DataFrame] = None
        self.indi = indi

    def load_and_filter_data(self, source, cluster) -> pd.DataFrame:
        df = pd.read_csv(self._csv_path)
      
        self._filtered_df = df[(df['source'] == source) & (df['cluster'] == cluster)]
       
        return self._filtered_df

    def count_images(self, source, cluster) -> pd.DataFrame:
        if self._filtered_df is None:
            self.load_and_filter_data(source, cluster)
        image_counts = []
        for id_name in self._filtered_df['id_name']:
            folder_path = os.path.join(self._parent_dir, id_name)
            if not os.path.exists(folder_path):
                image_counts.append({'id_name': id_name, 'image_count': 0})
                continue
            image_files = []
            for ext in ['jpg', 'jpeg', 'png', 'bmp', 'tiff']:
                image_files += glob(os.path.join(folder_path, f'*.{ext}')) + glob(os.path.join(folder_path, f'*.{ext.upper()}'))
            image_counts.append({'id_name': id_name, 'image_count': len(image_files)})
        self._results_df = pd.DataFrame(image_counts)

        self._results_df = self._results_df.merge(
            self._filtered_df[['id_name', 'Median RGB']],
            on='id_name',
            how='left'
        )
        return self._results_df

   

    def separate_low_count_folders(self, source, cluster) -> Tuple[float, float, pd.DataFrame, pd.DataFrame]:
        if self._results_df is None:
            self.count_images(source, cluster)
        counts = self._results_df['image_count'].values

        # Edge case: Only one folder
        if len(counts) == 1:
            # No clustering possible, no outliers possible
            stat_threshold = counts[0]  # or set to 0 or np.nan as appropriate
            kmeans_threshold = counts[0]
            self._low_count_stats = pd.DataFrame()  # empty
            self._low_count_kmeans = pd.DataFrame()  # empty
            return stat_threshold, kmeans_threshold, self._low_count_kmeans, self._results_df

        # Normal case: multiple folders
        q1 = np.percentile(counts, 25)
        q3 = np.percentile(counts, 75)
        iqr = q3 - q1
        stat_threshold = q1 - 1.5 * iqr

        # Only run KMeans if enough samples
        if len(counts) >= 2:
            kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
            clusters = kmeans.fit_predict(counts.reshape(-1, 1))
            centers = sorted(kmeans.cluster_centers_.flatten())
            kmeans_threshold = np.mean(centers)
        else:
            kmeans_threshold = stat_threshold  # fallback

        self._low_count_stats = self._results_df[self._results_df['image_count'] < stat_threshold]
        self._low_count_kmeans = self._results_df[self._results_df['image_count'] < kmeans_threshold]

        # Optionally, handle the case where no folders are classified as low count
        if self._low_count_stats.empty and self._low_count_kmeans.empty:
            # No outliers detected; handle as needed (e.g., log, return empty, etc.)
            pass

        return stat_threshold, kmeans_threshold, self._low_count_kmeans, self._results_df


    def get_low_count_kmeans(self) -> pd.DataFrame:
        return self._low_count_kmeans

    def get_results_df(self) -> pd.DataFrame:
        return self._results_df

class ImageMatcher:
    """Handles SIFT descriptor extraction and image matching."""
    
    def __init__(self, min_match_count: int = 10, use_flann: bool = True):
        self.min_match_count = min_match_count
        self.sift = cv2.SIFT_create(
            nfeatures=0,
            nOctaveLayers=5,
            contrastThreshold=0.02,
            edgeThreshold=15,
            sigma=1.2
        )
        self.use_flann = use_flann
        if self.use_flann:
            flann_index_kdtree = 1
            index_params = dict(algorithm=flann_index_kdtree, trees=5)
            search_params = dict(checks=50)
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    def compute_descriptors(self, img: np.ndarray) -> Optional[np.ndarray]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, descriptors = self.sift.detectAndCompute(gray, None)
        return descriptors

    def match_descriptors(self, des1: np.ndarray, des2: np.ndarray) -> int:
        if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
            return 0
        if self.use_flann:
            if len(des2) < 2:
                return 0
            des1 = des1.astype(np.float32)
            des2 = des2.astype(np.float32)
            matches = self.matcher.knnMatch(des1, des2, k=2)
            good = [m for m, n in matches if len([m, n]) == 2 and m.distance < 0.7 * n.distance]
            return len(good)
        else:
            matches = self.matcher.match(des1, des2)
            good = [m for m in matches if m.distance < 300]
            return len(good)

class FolderProcessor:
    """Loads and caches images and their descriptors for a folder."""
    
    def __init__(self, base_path: str):
        self.base_path = base_path
        self._folder_cache: Dict[str, Dict] = {}

    def load_folder(self, folder_id: str) -> Dict:
        if folder_id in self._folder_cache:
            return self._folder_cache[folder_id]
        folder_path = os.path.join(self.base_path, folder_id)
        image_paths = glob(os.path.join(folder_path, '*'))
        images, descriptors = [], []
        for img_path in image_paths:
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
                descriptors.append(None)
        self._folder_cache[folder_id] = {
            'images': images,
            'descriptors': descriptors,
            'path': folder_path
        }
        return self._folder_cache[folder_id]

    def compute_descriptors_for_folder(self, folder_id: str, matcher: ImageMatcher) -> Dict:
        folder_data = self._folder_cache.get(folder_id) or self.load_folder(folder_id)
        for i, img in enumerate(folder_data['images']):
            if folder_data['descriptors'][i] is None:
                folder_data['descriptors'][i] = matcher.compute_descriptors(img)
        return folder_data

class MatchingPipeline:
    """Coordinates the matching process between two sets of image folders."""
    
    def __init__(self, indi: str, base: str, csv_path: str, broadcast_base: str, tactimian_base: str,
                 source: str, cluster: int, min_match: int = 2, max_workers: int = 4):
        self.csv_path = csv_path
        self.broadcast_base = broadcast_base
        self.tactimian_base = tactimian_base
        self.min_match = min_match
        self.max_workers = max_workers
        self.matcher = ImageMatcher(min_match_count=min_match, use_flann=True)
        self.base = base
        self.indi = indi
        if self.indi == 'homo':
            self.broadcast_processor = FolderProcessor(self.base)
            self.tactimian_processor = FolderProcessor(self.base)
        else:
            self.broadcast_processor = FolderProcessor(self.broadcast_base)
            self.tactimian_processor = FolderProcessor(self.tactimian_base)
        self.src_ids: List[str] = []
        self.tgt_ids: List[str] = []
        self.detailed_results: List[Dict] = []
        self.summary_rows: List[Dict] = []
        self.source = source
        self.cluster = cluster

    def load_data_homo(self):
        analyzer = ImageFolderAnalyzer(self.csv_path, self.base, self.indi)
        _, _, df2, df1 = analyzer.separate_low_count_folders(self.source, self.cluster)
        merged = pd.merge(df1, df2[['id_name']], on='id_name', how='left', indicator=True)
        df3 = merged[merged['_merge'] == 'left_only'].drop('_merge', axis=1).reset_index(drop=True)
        self.src_ids = df2['id_name'].tolist()
        self.tgt_ids = df3['id_name'].tolist()

    def load_data_hetro(self):
        analyzer_src = ImageFolderAnalyzer(self.csv_path, self.broadcast_base, self.indi)
        analyzer_tgt = ImageFolderAnalyzer(self.csv_path, self.tactimian_base, self.indi)
        _, _, df2_src, df1_src = analyzer_src.separate_low_count_folders('first', self.cluster)
        _, _, df2_tgt, df1_tgt = analyzer_tgt.separate_low_count_folders('second', self.cluster)

        # For source
        if df2_src.empty:
            df3_src = df1_src
        else:
            merged_src = pd.merge(df1_src, df2_src[['id_name']], on='id_name', how='left', indicator=True)
            df3_src = merged_src[merged_src['_merge'] == 'left_only'].drop('_merge', axis=1).reset_index(drop=True)

        # For target
        if df2_tgt.empty:
            df3_tgt = df1_tgt
        else:
            merged_tgt = pd.merge(df1_tgt, df2_tgt[['id_name']], on='id_name', how='left', indicator=True)
            df3_tgt = merged_tgt[merged_tgt['_merge'] == 'left_only'].drop('_merge', axis=1).reset_index(drop=True)

        self.src_ids = df3_src['id_name'].tolist()
        self.tgt_ids = df3_tgt['id_name'].tolist()


    def process_folder_pair(self, src: str, tgt: str) -> Dict:
        src_data = self.broadcast_processor.compute_descriptors_for_folder(src, self.matcher)
        tgt_data = self.tactimian_processor.compute_descriptors_for_folder(tgt, self.matcher)
        matches = []
        for src_des in src_data['descriptors']:
            if src_des is None or len(src_des) == 0:
                continue
            for tgt_des in tgt_data['descriptors']:
                if tgt_des is None or len(tgt_des) == 0:
                    continue
                mc = self.matcher.match_descriptors(src_des, tgt_des)
                if mc >= self.matcher.min_match_count:
                    matches.append(mc)
        return {
            'source_id': src,
            'target_id': tgt,
            'matches': matches,
            'total_matches': len(matches),
            'max_match_count': max(matches) if matches else 0
        }

    def run_comparisons(self):
        # Pre-cache all folders
        for src in tqdm(self.src_ids, desc="Loading source folders"):
            self.broadcast_processor.load_folder(src)
        for tgt in tqdm(self.tgt_ids, desc="Loading target folders"):
            self.tactimian_processor.load_folder(tgt)

        # Process in parallel
        tasks = [(src, tgt) for src in self.src_ids for tgt in self.tgt_ids]
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.process_folder_pair, src, tgt): (src, tgt) for src, tgt in tasks}
            for future in tqdm(as_completed(futures), total=len(tasks), desc="Processing folder pairs"):
                result = future.result()
                src = result['source_id']
                tgt = result['target_id']
                for mc in result['matches']:
                    self.detailed_results.append({
                        'source_id': src,
                        'target_id': tgt,
                        'match_count': mc
                    })
                self.summary_rows.append({
                    'source_id': src,
                    'target_id': tgt,
                    'total_matches': result['total_matches'],
                    'max_match_count': result['max_match_count']
                })

    def get_detailed_results_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.detailed_results)

    def get_summary_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.summary_rows)

def run_matching_pipeline(config: dict) -> Dict[int, pd.DataFrame]:
    """Run matching pipeline for all clusters and return detailed results"""
    results = {}
    for cluster in [0,1,2,3]:
        pipeline = MatchingPipeline(
            indi="hetro",
            base=None,
            csv_path=config['csv_path'],
            broadcast_base=config['broadcast_base'], # print(self._results_df)
            tactimian_base=config['tactimian_base'],
            source=None,
            cluster=cluster,
            min_match=config.get('min_match', 2),
            max_workers=config.get('max_workers', 4)
        )
        pipeline.load_data_hetro()
        pipeline.run_comparisons()
        results[cluster] = pipeline.get_detailed_results_df()
    return results
