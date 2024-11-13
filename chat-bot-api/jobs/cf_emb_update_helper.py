import client
import data as dt
import util as ut
from models import EntityEmb
import model as ml
import pandas as pd
from torch.utils.data import DataLoader
import data.dataset as ds
import torch
from models import UserInteraction
import logging


class CFEmbUpdateJobHelper:
    def __init__(self, ctx):
        self.ctx = ctx

    def get_interactions(self):
        api_client = client.RecChatBotV1ApiClient()
        return pd.DataFrame(api_client.interactions())

    def split_dataset(self, interactions_df):
        train_set, test_set = dt.interactions_train_test_split(
            interactions_df, n_min_interactions=20, test_size=0.05
        )

        # Generate sequences....
        user_sequencer = dt.Sequencer("user_id", "user_seq")
        item_sequencer = dt.Sequencer("item_id", "item_seq")

        train_set = user_sequencer.perform(train_set)
        train_set = item_sequencer.perform(train_set)

        test_set = user_sequencer.perform(test_set)
        test_set = item_sequencer.perform(test_set)
        return train_set, test_set

    def update_embeddings(self, model, train_set):
        [user_embeddings, item_embeddings] = model.embedding.feature_embeddings

        def to_entity_embs(df, seq_col, id_col, embeddings):
            seq_to_id = ut.to_dict(df, seq_col, id_col)
            return [
                EntityEmb(id=str(id), emb=embeddings[seq].tolist())
                for seq, id in seq_to_id.items()
            ]

        user_embs = train_set.pipe(
            to_entity_embs, "user_seq", "user_id", user_embeddings
        )
        self.ctx.users_cf_emb_repository.upsert_many(user_embs)

        item_embs = train_set.pipe(
            to_entity_embs, "item_seq", "item_id", item_embeddings
        )
        self.ctx.items_cf_emb_repository.upsert_many(item_embs)

    async def update_database(self, train_set, model):
        logging.info("Generate train_set")

        prediction_set = self.__to_prediction_dataset(train_set)

        logging.info("Build train_set")

        predictor = ml.ModulePredictor(model)

        logging.info("Predict ratings")

        predictions = predictor.predict_dl(self.__to_dataloader(prediction_set))

        logging.info("Save predicted ratings")

        prediction_set["predicted_rating"] = predictions

        logging.info("Create interactions")

        models = [
            UserInteraction(
                user_id=row["user_id"],
                item_id=row["item_id"],
                rating=row["predicted_rating"],
            )
            for _, row in prediction_set.iterrows()
        ]

        logging.info("UPSERT interaction into ")

        await self.ctx.pred_interactions_repository.upsert_many(models)

        logging.info("FINISH UPSERT")


    def __to_prediction_dataset(self, train_set):
        items_df = train_set[["item_id", "item_seq"]].drop_duplicates()
        items_df

        chatbot_user_ids = train_set[train_set["user_id"].str.contains("@")][
            ["user_id", "user_seq"]
        ].drop_duplicates()
        chatbot_user_ids
        if len(chatbot_user_ids) == 0:
            raise Exception(
                "Not found any chatbot user interaction. Go to rate movies into chatbot-api and then come back and retry this action"
            )

        prediction_set = chatbot_user_ids.merge(items_df, how="cross")
        return prediction_set.drop_duplicates()

    def __to_dataloader(self, df, batch_size=64, num_workers=24, pin_memory=True):
        def to_tensor(obs, device, columns):
            data = obs[columns]
            if type(data) == pd.DataFrame:
                data = data.values
            return torch.tensor(data).to(device)

        features_fn = lambda obs, device: to_tensor(
            obs, device, ["user_seq", "item_seq"]
        )
        target_fn = lambda obs, device: torch.zeros((len(obs),)).to(device)

        dataset = ds.RecSysDataset(
            dataset=df, transform=features_fn, target_transform=target_fn
        )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
