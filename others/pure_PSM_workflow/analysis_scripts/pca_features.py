"""Preserve the numerical PCA environment used by the archived controls."""

def fitted_features(psm, *, transform=False):
    import numpy as np
    from sklearn.decomposition import PCA
    from threadpoolctl import threadpool_limits
    # Float32 randomized SVD changes slightly with BLAS threading. Small changes
    # can change tree bins, so the archived 32-thread environment is explicit.
    with threadpool_limits(limits=32, user_api="blas"):
        estimator = PCA(n_components=200, random_state=0)
        if transform:
            estimator.fit(psm)
            features = estimator.transform(psm)
        else:
            features = estimator.fit_transform(psm)
    return features.astype(np.float32), estimator.components_, float(estimator.explained_variance_ratio_.sum())
