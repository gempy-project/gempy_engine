import numpy as np

from gempy_engine.modules.weights_cache.weights_cache_interface import WeightCache, get_default_cache_dir, generate_cache_key

example_weights = np.array([.2, .2, .4, .2])


def test_save_weights():
    cache_dir = get_default_cache_dir()
    cache = WeightCache(cache_dir)

    weights_key = generate_cache_key(
        name="sandstone",
        parameters={
            "shape": 1,
            "sum": np.arange(10),
        }
    )

    cache.store_weights(
        key=weights_key,
        weights=example_weights
    )


def test_load_weights():
    # Load weights
    cache_dir = get_default_cache_dir()
    cache = WeightCache(cache_dir)
    
    weights_key = generate_cache_key(
        name="sandstone",
        parameters={
            "shape": 1,
            "sum": np.arange(10),
        }
    )

    retrieved_weights = cache.load_weights(weights_key)
    print(retrieved_weights)


def test_vector_quantizing():
    from sklearn.cluster import KMeans
    import numpy as np

    # Example: Quantizing a set of 2D vectors (can be adapted for higher dimensions)
    data = np.random.rand(10000, 1)  # 100 2D vectors

    # Train KMeans to find centroids
    n_clusters = 10  # Number of centroids (tune this based on your data)
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(data)

    # Quantize data
    quantized = kmeans.predict(data)  # Each vector in 'data' is now represented by a cluster index

    # Retrieving the centroids
    centroids = kmeans.cluster_centers_

    # Example: Using centroids for reconstruction
    reconstructed_data = centroids[quantized]
    pass
