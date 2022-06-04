class Fn:
    @staticmethod
    def train(
        model,
        data_loader, 
        loss_fn,
        optimizers,
        device
    ):
        model.train()
        total_loss = 0

        for index, (features, target) in enumerate(data_loader):
            features, target = features.to(device), target.to(device)
            y = model(features)

            model.zero_grad()
            loss = loss_fn(y, target)
            loss.backward()
            [op.step() for op in optimizers]
            total_loss += loss.item()

        return total_loss / len(data_loader)