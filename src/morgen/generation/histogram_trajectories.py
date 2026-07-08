from datetime import timedelta

import polars as pl
import torch
from meds import DataSchema, LabelSchema
from meds_torchdata import MEDSPytorchDataset


def format_histogram_trajectories(
    dataset: MEDSPytorchDataset,
    generated_outputs: list[torch.LongTensor],
    histogram_metadata_df: pl.DataFrame,
    window_size_days: int,
) -> pl.DataFrame:
    """Transfomrs the generated outputs into a MEDS-like dataframe format of continued trajectories.

    Args:
        dataset: The dataset used for generation.
        generated_outputs: The generated outputs. This is formatted as a list of generated samples that should
            be of the same length as the dataframe.

    Returns:
        A polars dataframe containing the generated trajectories in a MEDS-like format.

    Raises:
        ValueError: If the passed dataset does not yield code info with values strictly either always
            occurring or never occurring.
    """
    output_schema = dataset.schema_df.select(
        DataSchema.subject_id_name, LabelSchema.prediction_time_name, dataset.LAST_TIME
    )

    data_df = pl.DataFrame(
        {
            "code": [
                subject_codes[subject_codes != 0].numpy()
                for batch_codes in generated_outputs
                for subject_codes in batch_codes
            ]
        }
    )
    # TODO: Add support for using dataset.LAST_TIME, as start of prediction time. We don't need it for now for these experiments.
    data_df = data_df.with_columns(output_schema.head(data_df.height)["subject_id", "prediction_time"])

    data_df = data_df.explode("code").join(
        histogram_metadata_df["histogram/vocab_index", "number_of_days", "gap_count", "token_type"],
        left_on="code",
        right_on="histogram/vocab_index",
        how="left",
        maintain_order="left",
    )
    data_df = data_df.with_columns(
        time_offset=data_df.group_by(["subject_id", "prediction_time"], maintain_order=True)
        .agg(pl.col("number_of_days").cum_sum().alias("number_of_days"))["number_of_days"]
        .explode()
        - pl.duration(hours=round(window_size_days * 24 / 2))
    )
    data_df = data_df.with_columns((pl.col("prediction_time") + pl.col("time_offset")).alias("time"))

    min_time_offset = data_df.filter(~pl.col("token_type").is_in(["ANCHOR", "CODE"]))["time_offset"].min()
    expected_time = timedelta(hours=round(window_size_days * 24 / 2))
    if not min_time_offset == expected_time:
        print(f"Expected time offset to be {expected_time}, but got {min_time_offset}")

    data_df = data_df[
        [
            DataSchema.subject_id_name,
            LabelSchema.prediction_time_name,
            "time",
            "code",
            "token_type",
            "gap_count",
        ]
    ]
    schema_dict = {
        "subject_id": pl.Int64,
        "prediction_time": pl.Datetime(time_unit="us"),
        "time": pl.Datetime(time_unit="us"),
        "code": pl.Int64,
        "token_type": pl.Categorical(ordering="physical"),
        "gap_count": pl.Int64,
    }
    return pl.DataFrame(data_df, schema=schema_dict)
