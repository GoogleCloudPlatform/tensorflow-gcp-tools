# Lint as: python3
# Copyright 2019 Google LLC. All Rights Reserved.
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
"""Tests for google3.cloud.ml.tfe.keras4gcp.cloud_model_run._remote."""

import json
import os
import tempfile
from unittest import mock
from absl import flags
import numpy as np
import tensorflow as tf

from tensorflow_enterprise_addons.cloud_fit import _client
from tensorflow_enterprise_addons.cloud_fit import _remote

# If input dataset is created outside tuner.search(),
# it requires eager execution even in TF 1.x.
if tf.version.VERSION.split('.')[0] == '1':
  tf.compat.v1.enable_eager_execution()

MIRRORED_STRATEGY_NAME = _remote.MIRRORED_STRATEGY_NAME
MULTI_WORKER_MIRRORED_STRATEGY_NAME = _remote.MULTI_WORKER_MIRRORED_STRATEGY_NAME

FLAGS = flags.FLAGS
FLAGS.remote_dir = 'test_remote_dir'
FLAGS.distribution_strategy = MULTI_WORKER_MIRRORED_STRATEGY_NAME


class _MockCallable(object):

  @classmethod
  def reset(cls):
    cls.mock_callable = mock.Mock()

  @classmethod
  def mock_call(cls):
    cls.mock_callable()


class CustomCallbackExample(tf.keras.callbacks.Callback):

  def on_train_begin(self, unused_logs):
    _MockCallable.mock_call()


class CloudRunTest(tf.test.TestCase):

  def setUp(self):
    super(CloudRunTest, self).setUp()
    self._image_uri = 'gcr.io/some_test_image:latest'
    self._project_id = 'test_project_id'
    self._remote_dir = tempfile.mkdtemp()
    self._output_dir = os.path.join(self._remote_dir, 'output')
    self._model = self._half_plus_one_model()
    self._x = [[9.], [10.], [11.]] * 10
    self._y = [[xi[0] / 2. + 6] for xi in self._x]
    self._dataset = tf.data.Dataset.from_tensor_slices((self._x, self._y))
    self._logs_dir = os.path.join(self._remote_dir, 'logs')
    self._fit_kwargs = {
        'x': self._dataset,
        'verbose': 2,
        'batch_size': 30,
        'epochs': 10,
        'callbacks': [tf.keras.callbacks.TensorBoard(log_dir=self._logs_dir)]
    }
    _client._serialize_assets(self._remote_dir, self._model, **self._fit_kwargs)
    os.environ['TF_CONFIG'] = json.dumps({
        'cluster': {
            'worker': ['localhost:9999', 'localhost:9999']
        },
        'task': {
            'type': 'worker',
            'index': 0
        }
    })

  def _half_plus_one_model(self):
    """Writes SavedModel to compute y = wx + 1, with w trainable."""
    inp = tf.keras.layers.Input(shape=(1,), dtype=tf.float32)
    times_w = tf.keras.layers.Dense(
        units=1,
        kernel_initializer=tf.keras.initializers.Constant([[0.5]]),
        kernel_regularizer=tf.keras.regularizers.l2(0.01),
        use_bias=False)
    plus_1 = tf.keras.layers.Dense(
        units=1,
        kernel_initializer=tf.keras.initializers.Constant([[1.0]]),
        bias_initializer=tf.keras.initializers.Constant([1.0]),
        trainable=False)
    outp = plus_1(times_w(inp))
    model = tf.keras.Model(inp, outp)

    model.compile(
        tf.keras.optimizers.SGD(0.002), 'mean_squared_error', run_eagerly=True)
    return model

  def test_run(self):
    _remote.run(self._remote_dir, MIRRORED_STRATEGY_NAME)
    self.assertGreaterEqual(len(tf.io.gfile.listdir(self._output_dir)), 1)
    self.assertGreaterEqual(len(tf.io.gfile.listdir(self._logs_dir)), 1)

    model = tf.keras.models.load_model(self._output_dir)

    # Test saved model load and works properly
    self.assertGreater(model.losses, np.array([0.0], dtype=np.float32))

  def test_custom_callback(self):
    # Setting up custom callback with mock calls
    _MockCallable.reset()

    self._fit_kwargs['callbacks'] = [CustomCallbackExample()]
    _client._serialize_assets(self._remote_dir, self._model, **self._fit_kwargs)

    # Verify callback function has not been called yet.
    _MockCallable.mock_callable.assert_not_called()

    _remote.run(self._remote_dir, MIRRORED_STRATEGY_NAME)
    # Verifying callback functions triggered properly
    _MockCallable.mock_callable.assert_called_once_with()


if __name__ == '__main__':
  tf.test.main()
