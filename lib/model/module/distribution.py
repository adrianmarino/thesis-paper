import torch


class DistributionFactory:
    @staticmethod
    def normal(mu=0, sigma=1):
        dist = torch.distributions.Normal(mu, sigma)

        if torch.cuda.is_available():
            # Hack to get sampling on the GPU
            dist.loc   = dist.loc.cuda()
            dist.scale = dist.scale.cuda()
        return dist