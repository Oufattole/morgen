"""Lightning datamodule for histogram datasets.

Key differences vs MEDS base Datamodule
---------------------------------------
- Mirrors meds_torchdata.extensions.lightning_datamodule.Datamodule behavior:
  - Datasets are cached (single instance per split) using cached_property.
  - The DataLoader binds ``collate_fn=dataset.collate`` for the same dataset instance passed in.
    This avoids pickling a different dataset instance via a bound method in worker processes.
"""

from functools import cached_property

import lightning as L
from meds import held_out_split, train_split, tuning_split
from meds_torchdata.config import MEDSTorchDataConfig
from meds_torchdata.histogram_dataset import (
    EmptyWindowMode,
    FusionMode,
    HistogramConfig,
    HistogramPytorchDataset,
    WindowAnchoringStrategy,
)
from meds_torchdata.types import WindowSamplingStrategy
from torch.utils.data import DataLoader


class HistogramDatamodule(L.LightningDataModule):
    """A lightning datamodule for HistogramPytorchDataset.

    Similar to the regular MEDS datamodule but uses HistogramPytorchDataset which returns
    MEDSTorchHistogramBatch instead of MEDSTorchBatch.
    """

    def __init__(
        self,
        config: MEDSTorchDataConfig,
        batch_size: int = 32,
        num_workers: int | None = None,
        pin_memory: bool | None = None,
        persistent_workers: bool | None = None,
        prefetch_factor: int | None = None,
        anchoring_strategy: WindowAnchoringStrategy | str = WindowAnchoringStrategy.END_OF_TIMELINE,
        empty_window_mode: EmptyWindowMode | str = EmptyWindowMode.SINGLE_GAP,
        specific_event_regex: str | None = None,
        include_anchor_token: bool = False,
        histogram_max_seq_len: int = 512,
        fusion_mode: FusionMode = FusionMode.HISTOGRAM_ONLY,
        max_pre_anchor_tokens: int | None = None,
        max_code_seq_len: int | None = None,
        window_size_days: int | None = None,
        max_windows: int | None = None,
        window_sampling_strategy: WindowSamplingStrategy = WindowSamplingStrategy.RANDOM,
    ):
        super().__init__()
        self.config = config
        self.batch_size = batch_size
        self.num_workers = num_workers if num_workers is not None else 0
        self.pin_memory = pin_memory if pin_memory is not None else False
        self.persistent_workers = persistent_workers if num_workers > 0 else False
        self.prefetch_factor = prefetch_factor if num_workers > 0 else None

        if isinstance(anchoring_strategy, str):
            anchoring_strategy = WindowAnchoringStrategy[anchoring_strategy]

        if isinstance(empty_window_mode, str):
            empty_window_mode = EmptyWindowMode[empty_window_mode]
        
        if window_size_days is None:
            raise ValueError("window_size_days must be provided")

        # Create histogram configuration for autoregressive training
        self.histogram_config = HistogramConfig(
            window_sampling_strategy=window_sampling_strategy,
            max_windows=max_windows,
            window_size_days=window_size_days,
            anchoring_strategy=anchoring_strategy,  # Use actual enum object
            empty_window_mode=empty_window_mode,  # For autoregressive training
            empty_window_token=-1,
            vocab_size=config.vocab_size,
            padding_side=config.padding_side,
            specific_event_regex=specific_event_regex,
            include_anchor_token=include_anchor_token,
            histogram_max_seq_len=histogram_max_seq_len,
            fusion_mode=fusion_mode,
            max_pre_anchor_tokens=max_pre_anchor_tokens,
            max_code_seq_len=max_code_seq_len,
        )

    @cached_property
    def train_dataset(self) -> HistogramPytorchDataset:
        """Training dataset."""
        return HistogramPytorchDataset(self.config, train_split, self.histogram_config)

    @cached_property
    def val_dataset(self) -> HistogramPytorchDataset:
        """Validation dataset (tuning split)."""
        return HistogramPytorchDataset(self.config, tuning_split, self.histogram_config)

    @cached_property
    def test_dataset(self) -> HistogramPytorchDataset:
        """Test dataset (held_out split)."""
        return HistogramPytorchDataset(self.config, held_out_split, self.histogram_config)

    @property
    def shared_dataloader_kwargs(self) -> dict:
        """Shared kwargs for all dataloaders."""
        return {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers,
            "prefetch_factor": self.prefetch_factor,
        }

    def _dataloader(self, dataset: HistogramPytorchDataset, **kwargs) -> DataLoader:
        # Bind collate to the same dataset instance the DataLoader will use.
        return DataLoader(dataset, collate_fn=dataset.collate, **self.shared_dataloader_kwargs, **kwargs)

    def train_dataloader(self, shuffle: bool = True) -> DataLoader:
        """Training dataloader."""
        return self._dataloader(self.train_dataset, shuffle=shuffle)

    def val_dataloader(self) -> DataLoader:
        """Validation dataloader."""
        return self._dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        """Test dataloader."""
        return self._dataloader(self.test_dataset, shuffle=False)
