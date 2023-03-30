from torch.utils.data import DataLoader
import numpy as np


class LatentSpaceEncoder:
    def __init__(
        self,
        model,
        batch_size  = 100,
        num_workers = 24,
        pin_memory  = True
    ):
        self.model       = model
        self.batch_size  = batch_size
        self.num_workers = num_workers
        self.pin_memory  = pin_memory


    def __call__(self, data):
        dl = DataLoader(
            data,
            self.batch_size,
            num_workers = self.num_workers,
            pin_memory  = self.pin_memory
        )

        vectors = []
        for (x, _) in dl:
            z = self.model.encoder.predict(x)
            z = z.to('cpu').detach().numpy()
            vectors.append(z)

        return np.vstack(vectors)