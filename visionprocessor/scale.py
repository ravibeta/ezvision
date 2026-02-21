import numpy as np
from sklearn.decomposition import PCA

# Example batch of 1024-dimensional vectors
vectors_1024 = np.random.rand(100, 1024)  # Replace with actual API output

# Expand to 1536 dimensions using PCA
pca = PCA(n_components=1536)
vectors_1536 = pca.fit_transform(vectors_1024)

print(f"Expanded Vector Shape: {vectors_1536.shape}")  # Should be (100, 1536)
