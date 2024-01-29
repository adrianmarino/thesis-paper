import util as ut
from models import EntityEmb


def to_entity_embs(df, seq_col, id_col, embeddings):
    seq_to_id = ut.to_dict(df, seq_col, id_col)
    return [
        EntityEmb(
            id  = id,
            emb = embeddings[seq].tolist()
        )
        for seq, id in seq_to_id.items()
    ]