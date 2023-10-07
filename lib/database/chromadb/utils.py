from bunch import Bunch


def to_insert_params(
    df,
    id_col='id',
    metadata_cols=[],
    embedding_col=None,
    text_col=None
):
    ids = df[id_col].astype(str).values.tolist()

    def get_str(row, column):
        value = row[column] if column in row else None
        return value.strip() if type(value) == str else value

    metadatas = [{c: get_str(row, c) for c in metadata_cols} for _, row in df.iterrows()] \
        if len(metadata_cols) > 0 else None

    embeddings = df[embedding_col].values.tolist() if embedding_col else None
    documents  = df[text_col].str.strip().values.tolist() if text_col else None

    return Bunch({
        'ids'        : ids,
        'metadatas'  : metadatas,
        'embeddings' : embeddings,
        'documents'  : documents
    })
