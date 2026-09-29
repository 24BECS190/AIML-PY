import numpy as np


def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


class KMeans:
    def __init__(self, k=2, max_iters=100):
        self.k = k
        self.max_iters = max_iters

    def fit(self, X):
        # Pick random initial centroids from the dataset
        idx = np.random.choice(
            X.shape[0],
            self.k,
            replace=False
        )

        self.centroids = X[idx].astype(float)

        for _ in range(self.max_iters):

            # Assign clusters
            self.labels = np.array([
                np.argmin([
                    euclidean_distance(x, c)
                    for c in self.centroids
                ])
                for x in X
            ])

            # Recompute centroids
            new_centroids = np.array([
                X[self.labels == i].mean(axis=0)
                if len(X[self.labels == i]) > 0
                else self.centroids[i]
                for i in range(self.k)
            ])

            # Check convergence
            if np.allclose(
                self.centroids,
                new_centroids
            ):
                break

            self.centroids = new_centroids

        return self.labels


def silhouette_score_simple(X, labels):
    scores = []

    for i, x in enumerate(X):
        c_idx = labels[i]

        same_cluster = X[labels == c_idx]
        other_clusters = set(labels) - {c_idx}

        # Calculate a: mean distance within same cluster
        if len(same_cluster) > 1:
            distances = [
                euclidean_distance(x, point)
                for j, point in enumerate(same_cluster)
                if not np.array_equal(x, point)
            ]

            a = np.mean(distances) if distances else 0
        else:
            a = 0

        # Calculate b: minimum mean distance to another cluster
        if other_clusters:
            cluster_distances = []

            for c in other_clusters:
                other_cluster = X[labels == c]

                mean_distance = np.mean([
                    euclidean_distance(x, point)
                    for point in other_cluster
                ])

                cluster_distances.append(mean_distance)

            b = min(cluster_distances)
        else:
            b = 0

        # Calculate silhouette value
        if max(a, b) == 0:
            scores.append(0)
        else:
            scores.append((b - a) / max(a, b))

    return np.mean(scores)


# Example Usage
if __name__ == "__main__":

    X = np.array([
        [1, 2],
        [1, 4],
        [1, 0],
        [10, 2],
        [10, 4],
        [10, 0]
    ])

    km = KMeans(k=2)

    labels = km.fit(X)

    score = silhouette_score_simple(X, labels)

    print("Assigned Labels:", labels)
    print("Calculated Silhouette Validation Score:", score)