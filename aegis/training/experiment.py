"""
AEGIS Epoch Experiment Runner.

Version: 0.18.0
"""

from pathlib import Path

from .checkpoint import (
    save_checkpoint,
)

from .early_stopping import (
    EarlyStoppingState,
)

from .epoch import (
    run_training_epoch,
    run_validation_epoch,
)

from .history import (
    EpochRecord,
    ExperimentHistory,
)

from .trainer import (
    AEGISTrainer,
)


class AEGISExperimentRunner:
    """
    Runs complete AEGIS training experiments.

    Responsibilities:
    - epoch training
    - validation
    - best-model tracking
    - early stopping
    - checkpointing
    - experiment history
    """

    def __init__(
        self,
        trainer: AEGISTrainer,
        experiment_name: str = (
            "aegis_experiment"
        ),
        max_epochs: int = 10,
        patience: int = 5,
        min_delta: float = 0.0,
        checkpoint_directory=(
            "checkpoints"
        ),
        save_best: bool = True,
        save_last: bool = True,
    ):

        if not isinstance(
            trainer,
            AEGISTrainer,
        ):
            raise TypeError(
                "trainer must be AEGISTrainer."
            )

        if max_epochs < 1:
            raise ValueError(
                "max_epochs must be at least 1."
            )

        if not experiment_name:
            raise ValueError(
                "experiment_name cannot be empty."
            )

        self.trainer = trainer

        self.experiment_name = (
            experiment_name
        )

        self.max_epochs = (
            max_epochs
        )

        self.save_best = (
            save_best
        )

        self.save_last = (
            save_last
        )

        self.checkpoint_directory = (
            Path(
                checkpoint_directory
            )
            / experiment_name
        )

        self.early_stopping = (
            EarlyStoppingState(
                patience=patience,
                min_delta=min_delta,
            )
        )

        self.history = (
            ExperimentHistory(
                experiment_name=(
                    experiment_name
                )
            )
        )

    @property
    def best_checkpoint_path(
        self,
    ):

        return (
            self.checkpoint_directory
            / "best.pt"
        )

    @property
    def last_checkpoint_path(
        self,
    ):

        return (
            self.checkpoint_directory
            / "last.pt"
        )

    @property
    def history_path(
        self,
    ):

        return (
            self.checkpoint_directory
            / "history.json"
        )

    def _save_checkpoint(
        self,
        path,
        epoch,
        metrics,
        checkpoint_type,
    ):

        return save_checkpoint(
            path=path,

            alignment_model=(
                self.trainer
                .alignment_model
            ),

            classification_model=(
                self.trainer
                .classification_model
            ),

            optimizer=(
                self.trainer
                .optimizer
            ),

            epoch=epoch,

            global_step=(
                self.trainer
                .state
                .global_step
            ),

            metrics=metrics,

            extra={
                "experiment_name": (
                    self.experiment_name
                ),

                "checkpoint_type": (
                    checkpoint_type
                ),
            },
        )

    def run(
        self,
        train_loader,
        validation_loader,
    ) -> ExperimentHistory:
        """
        Run the complete AEGIS experiment.
        """

        self.checkpoint_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        for epoch in range(
            1,
            self.max_epochs + 1,
        ):

            train_metrics = (
                run_training_epoch(
                    trainer=(
                        self.trainer
                    ),
                    dataloader=(
                        train_loader
                    ),
                )
            )

            validation_metrics = (
                run_validation_epoch(
                    trainer=(
                        self.trainer
                    ),
                    dataloader=(
                        validation_loader
                    ),
                )
            )

            validation_loss = (
                validation_metrics[
                    "loss"
                ]
            )

            improved = (
                self.early_stopping
                .update(
                    validation_loss
                )
            )

            self.trainer.state.epoch = (
                epoch
            )

            if improved:

                self.trainer.state.best_validation_loss = (
                    validation_loss
                )

                if self.save_best:

                    self._save_checkpoint(
                        path=(
                            self.best_checkpoint_path
                        ),

                        epoch=epoch,

                        metrics=(
                            validation_metrics
                        ),

                        checkpoint_type=(
                            "best"
                        ),
                    )

            record = EpochRecord(
                epoch=epoch,

                train_metrics=(
                    train_metrics
                ),

                validation_metrics=(
                    validation_metrics
                ),

                improved=(
                    improved
                ),
            )

            self.history.add(
                record
            )

            if (
                self.early_stopping
                .stopped
            ):

                self.history.stopped_early = (
                    True
                )

                break

        if self.save_last:

            final_record = (
                self.history.records[
                    -1
                ]
            )

            self._save_checkpoint(
                path=(
                    self.last_checkpoint_path
                ),

                epoch=(
                    final_record.epoch
                ),

                metrics=(
                    final_record
                    .validation_metrics
                ),

                checkpoint_type=(
                    "last"
                ),
            )

        self.history.save(
            self.history_path
        )

        return self.history