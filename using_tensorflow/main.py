
import numpy as np 
import logging, sys 

import tensorflow as tf 
import optuna
from optuna.integration import TFKerasPruningCallback
from optuna.samplers import TPESampler, RandomSampler
from optuna.pruners import BasePruner
from optuna.trial._state import TrialState

import model_file 
import dataset_file


def objective(trial):
    # Clear clutter from previous TensorFlow graphs.
    tf.keras.backend.clear_session()

    # Metrics to be monitored by Optuna.
    if tf.__version__ >= "2":
        monitor = "val_accuracy"
    else:
        monitor = "val_acc"

    # Create tf.keras model instance.
    model = model_file.create_model(trial)

    # Create dataset instance.
    ds_train = dataset_file.train_dataset()
    ds_eval = dataset_file.eval_dataset()

    # Create callbacks for early stopping and pruning.
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3),
        TFKerasPruningCallback(trial, monitor),
    ]

    # Train model.
    history = model.fit(
        ds_train,
        epochs=20,
        steps_per_epoch=len(ds_train),
        validation_data=ds_eval,
        validation_steps=len(ds_eval),
        callbacks=callbacks
    )

    return history.history[monitor][-1]


def show_result(study, df):

    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    
    # experiment print
    print(df)


class SimulatedAnnealingSampler(optuna.samplers.BaseSampler):
    def __init__(self, temperature=100):
        self._rng = np.random.RandomState()
        self._temperature = temperature  # Current temperature.
        self._current_trial = None  # Current state.

    def sample_relative(self, study, trial, search_space):
        if search_space == {}:
            return {}

        # Simulated Annealing algorithm.
        # 1. Calculate transition probability.
        prev_trial = study.trials[-2]
        if self._current_trial is None or prev_trial.value <= self._current_trial.value:
            probability = 1.0
        else:
            probability = np.exp(
                (self._current_trial.value - prev_trial.value) / self._temperature
            )
        self._temperature *= 0.9  # Decrease temperature.

        # 2. Transit the current state if the previous result is accepted.
        if self._rng.uniform(0, 1) < probability:
            self._current_trial = prev_trial

        # 3. Sample parameters from the neighborhood of the current point.
        # The sampled parameters will be used during the next execution of
        # the objective function passed to the study.
        params = {}
        for param_name, param_distribution in search_space.items():
            if not isinstance(param_distribution, optuna.distributions.UniformDistribution):
                raise NotImplementedError("Only suggest_float() is supported")

            current_value = self._current_trial.params[param_name]
            width = (param_distribution.high - param_distribution.low) * 0.1
            neighbor_low = max(current_value - width, param_distribution.low)
            neighbor_high = min(current_value + width, param_distribution.high)
            params[param_name] = self._rng.uniform(neighbor_low, neighbor_high)

        return params

    # The rest are unrelated to SA algorithm: boilerplate
    def infer_relative_search_space(self, study, trial):
        return optuna.samplers.intersection_search_space(study)

    def sample_independent(self, study, trial, param_name, param_distribution):
        independent_sampler = optuna.samplers.RandomSampler()
        return independent_sampler.sample_independent(study, trial, param_name, param_distribution)
    
class LastPlacePruner(BasePruner):
    def __init__(self, warmup_steps, warmup_trials):
        self._warmup_steps = warmup_steps
        self._warmup_trials = warmup_trials

    def prune(self, study: "optuna.study.Study", trial: "optuna.trial.FrozenTrial") -> bool:
        # Get the latest score reported from this trial
        step = trial.last_step

        if step:  # trial.last_step == None when no scores have been reported yet
            this_score = trial.intermediate_values[step]

            # Get scores from other trials in the study reported at the same step
            completed_trials = study.get_trials(deepcopy=False)
            other_scores = [
                t.intermediate_values[step]
                for t in completed_trials
                if step in t.intermediate_values
            ]
            other_scores = sorted(other_scores)

            # Prune if this trial at this step has a lower value than all completed trials
            # at the same step. Note that steps will begin numbering at 0 in the objective
            # function definition below.
            if step >= self._warmup_steps and len(other_scores) > self._warmup_trials:
                if this_score < other_scores[0]:
                    print(f"prune() True: Trial {trial.number}, Step {step}, Score {this_score}")
                    return True

        return False


def main():
    # Add stream handler of stdout to show the messages
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = "covid19_image"  # Unique identifier of the study.
    storage_name = "sqlite:///{}.db".format(study_name)
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        sampler=SimulatedAnnealingSampler(),
        direction="maximize", 
        pruner=LastPlacePruner(warmup_steps=1, warmup_trials=5),
        #pruner=optuna.pruners.MedianPruner(n_startup_trials=2),
        load_if_exists=True
    )
    
    # training
    study.optimize(objective, n_trials=1, timeout=600)
    # experimental history
    df = study.trials_dataframe(attrs=("number", "value", "params", "state"))

    show_result(study, df)


if __name__ == "__main__":
    main()

