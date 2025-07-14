import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import ast
import numpy as np
from color_analysis import TrackedObjectsAnalyzer, SaliencyColorClassifier

root_folder_1 = "/home/levi/Documents/pipecat/pipecat_quick_Start/broadcast_object-20250713T171653Z-1-001/broadcast_object"
analyzer_1 = TrackedObjectsAnalyzer(root_folder_1, SaliencyColorClassifier)
analyzer_1.analyze_all()

# Save the summary as a DataFrame in a variable
df1 = analyzer_1.to_dataframe()
# print(df1)

root_folder_2 = "/home/levi/Documents/pipecat/pipecat_quick_Start/tactimian_object-20250713T171700Z-1-001/tactimian_object"
analyzer_2 = TrackedObjectsAnalyzer(root_folder_2, SaliencyColorClassifier)
analyzer_2.analyze_all()

# Save the summary as a DataFrame in a variable
df2 = analyzer_2.to_dataframe()
# print(df2)
# Now you can use summary_df as a normal pandas DataFrame



# Load both datasets with source identifiers

df1['source'] = 'first'  # Add source column


df2['source'] = 'second'  # Add source column

df1.rename(columns={'median_rgb': 'Median RGB'}, inplace=True)
df2.rename(columns={'median_rgb': 'Median RGB'}, inplace=True)
# Combine datasets
combined_df = pd.concat([df1, df2], ignore_index=True)

# Function to parse RGB tuples safely
def parse_rgb(rgb_str):
    try:
        # Handle different string formats
        cleaned = rgb_str.replace("np.int64", "").replace("(", "").replace(")", "")
        return tuple(map(int, ast.literal_eval(cleaned)))
    except:
        return (0, 0, 0)  # Default for parsing errors

# Parse median RGB values
# combined_df['Median RGB'] = combined_df['Median RGB'].apply(parse_rgb)
print(combined_df)
# Prepare data for clustering
median_colors = np.array(combined_df['Median RGB'].tolist())

# Apply KMeans clustering
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
combined_df['cluster'] = kmeans.fit_predict(median_colors)

# Create display IDs with source prefixes
combined_df['display_id'] = combined_df['source'] + '_' + combined_df['id_name'].astype(str)


# Save final results to CSV
combined_df[['id_name', 'source', 'cluster', 'Median RGB']].to_csv('final_new.csv', index=False)
print("Saved cluster assignments to final.csv")

# # Plot clustered colors with prefixed IDs
# fig, axs = plt.subplots(nrows=n_clusters, figsize=(12, n_clusters * 2))

# for cluster_id in range(n_clusters):
#     cluster_df = combined_df[combined_df['cluster'] == cluster_id]
#     cluster_colors = cluster_df['Median RGB'].tolist()
#     display_ids = cluster_df['display_id'].tolist()
#     n_colors = len(cluster_colors)

#     if n_colors == 0:
#         print(f"Cluster {cluster_id} is empty — skipping plot")
#         continue

#     img = np.zeros((20, n_colors * 40, 3), dtype=np.uint8)


#     # Create image representation
   
#     for i, color in enumerate(cluster_colors):
#         img[:, i*40:(i+1)*40, :] = color

#     # Plot with annotations
     
#     axs[cluster_id].imshow(img)
#     axs[cluster_id].axis('off')
#     axs[cluster_id].set_title(f"Cluster {cluster_id}")

#     # Add source-prefixed IDs below each color
#     for i, did in enumerate(display_ids):
#         axs[cluster_id].text(
#             i*40 + 20,
#             30,
#             did,
#             fontsize=8,
#             ha='center',
#             va='top',
#             rotation=90
#         )

# plt.tight_layout()
# plt.savefig('color_clusters_1.png', dpi=300)
# plt.show()

# Plot clustered colors with prefixed IDs
fig, axs = plt.subplots(nrows=n_clusters, figsize=(12, n_clusters * 2))

for cluster_id in range(n_clusters):
    cluster_df = combined_df[combined_df['cluster'] == cluster_id]
    cluster_colors = cluster_df['Median RGB'].tolist()
    display_ids = cluster_df['display_id'].tolist()
    n_colors = len(cluster_colors)

    # Create image representation
    img = np.zeros((20, n_colors * 40, 3), dtype=np.uint8)
    for i, color in enumerate(cluster_colors):
        img[:, i*40:(i+1)*40, :] = color

    # Plot with annotations
    axs[cluster_id].imshow(img)
    axs[cluster_id].axis('off')
    axs[cluster_id].set_title(f"Cluster {cluster_id}")

    # Add source-prefixed IDs below each color
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
plt.savefig('color_clusters_1.png', dpi=300)
