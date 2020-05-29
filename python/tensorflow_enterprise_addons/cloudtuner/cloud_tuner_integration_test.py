# Lint as: python3
# Copyright 2020 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Integration tests for Cloud Keras Tuner."""

import contextlib
import io
from multiprocessing import dummy
import os
import re
import time
import kerastuner
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
from tensorflow_enterprise_addons import cloudtuner

# The project id to use to run tests.
_PROJECT_ID = os.environ['CT_E2E_PROJECT_ID']

# The GCP region in which the end-to-end test is run.
_REGION = os.environ['CT_E2E_REGION']

# Study ID for testing
_STUDY_ID_BASE = os.environ['CT_E2E_STUDY_ID']

# The search space for hyperparameters
_HPS = kerastuner.engine.hyperparameters.HyperParameters()
_HPS.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
_HPS.Int('num_layers', 2, 10)

# Number of search loops we would like to run in parallel for distributed tuning
_NUM_PARALLEL_TRIALS = 4


def _normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label


class CloudTunerIntegrationTest(tf.test.TestCase):

  @classmethod
  def setUpClass(cls):
    super(CloudTunerIntegrationTest, cls).setUpClass()
    # Prepare data
    (x, y), (val_x, val_y) = keras.datasets.mnist.load_data()
    x = x.astype('float32') / 255.
    val_x = val_x.astype('float32') / 255.

    cls._x = x[:10000]
    cls._y = y[:10000]
    cls._val_x = val_x
    cls._val_y = val_y

  def setUp(self):
    super(CloudTunerIntegrationTest, self).setUp()
    self._input_shape = (28, 28)

  def _single_tuner_dist(self, study_id, tuner_id):
    """Instantiate a CloudTuner for distributed tuning and set up its tuner_id.

    Args:
      study_id: String.
      tuner_id: Integer.

    Returns:
      A CloudTuner instance.
    """
    tuner = cloudtuner.CloudTuner(
        self._build_model,
        project_id=_PROJECT_ID,
        region=_REGION,
        objective='acc',
        hyperparameters=_HPS,
        max_trials=5,
        study_id=study_id,
        directory=os.path.join(self.get_temp_dir(), study_id, str(tuner_id)))
    tuner.tuner_id = str(tuner_id)
    return tuner

  def _build_model(self, hparams):
    # Note that CloudTuner does not support adding hyperparameters in the model
    # building function. Instead, the search space is configured by passing a
    # hyperparameters argument when instantiating (constructing) the tuner.
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=self._input_shape))

    # Build the model with number of layers from the hyperparameters
    for _ in range(hparams.get('num_layers')):
      model.add(keras.layers.Dense(units=64, activation='relu'))
    model.add(keras.layers.Dense(10, activation='softmax'))

    # Compile the model with learning rate from the hyperparameters
    model.compile(
        optimizer=keras.optimizers.Adam(lr=hparams.get('learning_rate')),
        loss='sparse_categorical_crossentropy',
        metrics=['acc'])
    return model

  def _search_fn(self, tuner):
    # Start searching from different time points for each worker
    # to avoid model.build collision. If not spaced out, deadlocks might occur
    # due to the Global Interpreter Lock in Python, that prevents multiple
    # threads from executing Python bytecode at once.
    time.sleep(int(tuner.tuner_id) * 2)
    tuner.search(
        x=self._x,
        y=self._y,
        epochs=5,
        validation_data=(self._val_x, self._val_y),
        verbose=0)
    return tuner

  def _assert_output(self, fn, regex_str):
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
      fn()
    output = stdout.getvalue()
    self.assertRegex(output, re.compile(regex_str, re.DOTALL))

  def _assert_results_summary(self, fn):
    self._assert_output(fn,
                        '.*Results summary.*Trial summary.*Hyperparameters.*')

  def testCloudTunerHyperparameters(self):
    study_id = '{}_hyperparameters'.format(_STUDY_ID_BASE)
    tuner = cloudtuner.CloudTuner(
        self._build_model,
        project_id=_PROJECT_ID,
        region=_REGION,
        objective='acc',
        hyperparameters=_HPS,
        max_trials=5,
        study_id=study_id,
        directory=os.path.join(self.get_temp_dir(), study_id))

    # "Search space summary" comes first, but the order of
    # "learning_rate (Float)" and "num_layers (Int)" is not deterministic,
    # hence they are wrapped as look-ahead assertions in the regex.
    self._assert_output(
        tuner.search_space_summary,
        r'.*Search space summary(?=.*learning_rate \(Float\))'
        r'(?=.*num_layers \(Int\).*)')

    tuner.search(
        x=self._x,
        y=self._y,
        epochs=10,
        validation_data=(self._val_x, self._val_y))

    self._assert_results_summary(tuner.results_summary)

  def testCloudTunerDatasets(self):
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    ds_train = ds_train.map(
        _normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(128)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    ds_test = ds_test.map(
        _normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.batch(128)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    # The input shape of images loaded from TF Datasets is different from
    # that of the Keras Datasets.
    self._input_shape = (28, 28, 1)

    study_id = '{}_dataset'.format(_STUDY_ID_BASE)
    tuner = cloudtuner.CloudTuner(
        self._build_model,
        project_id=_PROJECT_ID,
        region=_REGION,
        objective='acc',
        hyperparameters=_HPS,
        study_id=study_id,
        max_trials=5,
        directory=os.path.join(self.get_temp_dir(), study_id))

    self._assert_output(
        tuner.search_space_summary,
        r'.*Search space summary(?=.*learning_rate \(Float\))'
        r'(?=.*num_layers \(Int\).*)')

    tuner.search(x=ds_train, epochs=10, validation_data=ds_test)

    self._assert_results_summary(tuner.results_summary)

  def testCloudTunerStudyConfig(self):
    # Configure the search space. Specification:
    # https://cloud.google.com/ai-platform/optimizer/docs/reference/rest/v1/projects.locations.studies#StudyConfig
    study_config = {
        'metrics': [{
            'goal': 'MAXIMIZE',
            'metric': 'acc'
        }],
        'parameters': [{
            'discrete_value_spec': {
                'values': [0.0001, 0.001, 0.01]
            },
            'parameter': 'learning_rate',
            'type': 'DISCRETE'
        }, {
            'integer_value_spec': {
                'max_value': 10,
                'min_value': 2
            },
            'parameter': 'num_layers',
            'type': 'INTEGER'
        }, {
            'discrete_value_spec': {
                'values': [32, 64, 96, 128]
            },
            'parameter': 'units',
            'type': 'DISCRETE'
        }],
        'algorithm': 'ALGORITHM_UNSPECIFIED',
        'automatedStoppingConfig': {
            'decayCurveStoppingConfig': {
                'useElapsedTime': True
            }
        }
    }

    study_id = '{}_study_config'.format(_STUDY_ID_BASE)
    tuner = cloudtuner.CloudTuner(
        self._build_model,
        project_id=_PROJECT_ID,
        region=_REGION,
        study_config=study_config,
        study_id=study_id,
        max_trials=5,
        directory=os.path.join(self.get_temp_dir(), study_id))

    self._assert_output(
        tuner.search_space_summary,
        r'.*Search space summary(?=.*learning_rate \(Choice\))'
        r'(?=.*num_layers \(Int\))(?=.*units \(Choice\))')

    tuner.search(
        x=self._x,
        y=self._y,
        epochs=5,
        steps_per_epoch=2000,
        validation_steps=1000,
        validation_data=(self._val_x, self._val_y))

    self._assert_results_summary(tuner.results_summary)

  def testCloudTunerDistributedTuning(self):
    study_id = '{}_dist'.format(_STUDY_ID_BASE)
    tuners = [
        self._single_tuner_dist(study_id, i)
        for i in range(_NUM_PARALLEL_TRIALS)
    ]
    pool = dummy.Pool(processes=_NUM_PARALLEL_TRIALS)
    results = pool.map(self._search_fn, tuners)
    pool.close()
    pool.join()

    self._assert_results_summary(results[0].results_summary)


if __name__ == '__main__':
  tf.test.main()
