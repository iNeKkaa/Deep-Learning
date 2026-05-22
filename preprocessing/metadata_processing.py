def choose_metadata_columns(dataframe):
    """
    Select simple static metadata features.

    This list is deliberately conservative: we avoid identifiers, text reports,
    file paths and label-related columns.
    """
    candidate_columns = ["age", "sex", "height", "weight"]
    return [column for column in candidate_columns if column in dataframe.columns]


def fit_metadata_scaler(train_df, metadata_columns):
    """Compute mean and standard deviation on the training split only."""
    if len(metadata_columns) == 0:
        return {"mean": {}, "std": {}}

    values = train_df[metadata_columns].copy()
    for column in metadata_columns:
        values[column] = values[column].astype("float32")

    means = values.mean(axis=0, skipna=True).fillna(0.0)
    stds = values.std(axis=0, skipna=True).replace(0.0, 1.0).fillna(1.0)

    return {"mean": means.to_dict(), "std": stds.to_dict()}


def transform_metadata(dataframe, metadata_columns, scaler):
    """Fill missing values and standardize metadata columns."""
    dataframe = dataframe.copy()
    for column in metadata_columns:
        mean = scaler["mean"][column]
        std = scaler["std"][column]
        dataframe[column] = dataframe[column].astype("float32")
        dataframe[column] = dataframe[column].fillna(mean)
        dataframe[column] = (dataframe[column] - mean) / std
    return dataframe
