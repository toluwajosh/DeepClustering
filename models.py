"""models for DEC
"""

import torch
from torch import nn
from torch.nn import Parameter

from matplotlib import pyplot as plt


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 500),
            nn.ReLU(True),
            nn.Linear(500, 500),
            nn.ReLU(True),
            nn.Linear(500, 500),
            nn.ReLU(True),
            nn.Linear(500, 2000),
            nn.ReLU(True),
            nn.Linear(2000, 10),
        )
        self.decoder = nn.Sequential(
            nn.Linear(10, 2000),
            nn.ReLU(True),
            nn.Linear(2000, 500),
            nn.ReLU(True),
            nn.Linear(500, 500),
            nn.ReLU(True),
            nn.Linear(500, 500),
            nn.ReLU(True),
            nn.Linear(500, 28 * 28),
        )
        self.model = nn.Sequential(self.encoder, self.decoder)

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        x = self.model(x)
        return x


class ClusteringLayer(nn.Module):
    def __init__(
        self, n_clusters=10, hidden=10, cluster_centers=None, alpha=1.0
    ):
        super(ClusteringLayer, self).__init__()
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.hidden = hidden
        if cluster_centers is None:
            initial_cluster_centers = torch.zeros(
                self.n_clusters, self.hidden, dtype=torch.float
            ).cuda()
            nn.init.xavier_uniform_(initial_cluster_centers)
        else:
            initial_cluster_centers = cluster_centers
        self.cluster_centers = Parameter(initial_cluster_centers)

    def forward(self, x):
        norm_squared = torch.sum(
            (x.unsqueeze(1) - self.cluster_centers) ** 2, 2
        )
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator ** power
        t_dist = (
            numerator.t() / torch.sum(numerator, 1)
        ).t()  # soft assignment using t-distribution
        return t_dist


class DEC(nn.Module):
    def __init__(
        self,
        n_clusters=10,
        autoencoder=None,
        hidden=10,
        cluster_centers=None,
        alpha=1.0,
    ):
        super(DEC, self).__init__()
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.hidden = hidden
        self.cluster_centers = cluster_centers
        self.autoencoder = autoencoder
        self.clusteringlayer = ClusteringLayer(
            self.n_clusters, self.hidden, self.cluster_centers, self.alpha
        )

    def target_distribution(self, q_):
        """Compute target distributions

        Args:
            q_ (torch tensor): sample to cluster similarity

        Returns:
            torch tensor: auxiliary target distribution
        """
        weight = (q_ ** 2) / torch.sum(q_, 0)
        return (weight.t() / torch.sum(weight, 1)).t()

    def forward(self, x):
        x = self.autoencoder.encode(x)
        return self.clusteringlayer(x)

    def visualize(self, epoch, x):
        fig = plt.figure()
        ax = plt.subplot(111)
        x = self.autoencoder.encode(x).detach()
        x = x.cpu().numpy()[:2000]
        x_embedded = TSNE(n_components=2).fit_transform(x)
        plt.scatter(x_embedded[:, 0], x_embedded[:, 1])
        fig.savefig("plots/mnist_{}.png".format(epoch))
        plt.close(fig)
