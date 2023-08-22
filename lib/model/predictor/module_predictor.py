from model.predictor.abstract_predictor import AbstractPredictor
import torch


class ModulePredictor(AbstractPredictor):
    def __init__(self, model): self.model = model


    @property
    def name(self): return str(self.model.__class__.__name__)


    def predict(self, user_idx, item_idx,  n_neighbors=10, debug=False):
        input_batch = torch.tensor([user_idx, item_idx]).unsqueeze(0)

        return self.predict_batch(input_batch)


    def predict_batch(self, batch, n_neighbors=10, debug=False):
        y_pred = self.model.predict(batch)

        return y_pred.cpu().detach()


    def predict_dl(self, data_loader, n_neighbors=10, debug=False):
        predictions = []
        for features, _ in data_loader:
            predictions.append(self.predict_batch(features, n_neighbors, debug))
        return torch.concat(predictions)
