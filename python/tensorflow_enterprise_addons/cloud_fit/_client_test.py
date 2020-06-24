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
"""Tests for cloud_fit._client."""

import os
import tempfile
from unittest import mock
import cloudpickle
from googleapiclient import discovery
import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow_enterprise_addons.cloud_fit import _client
from tensorflow_enterprise_addons.cloud_fit import cloud_fit_utils

# Can only export Datasets which were created executing eagerly
cloud_fit_utils.enable_eager_for_tf_1()

MIRRORED_STRATEGY_NAME = cloud_fit_utils.MIRRORED_STRATEGY_NAME
MULTI_WORKER_MIRRORED_STRATEGY_NAME = cloud_fit_utils.MULTI_WORKER_MIRRORED_STRATEGY_NAME


class CloudRunModelTest(tf.test.TestCase):

  def setUp(self):
    super(CloudRunModelTest, self).setUp()
    self._image_uri = 'gcr.io/some_test_image:latest'
    self._project_id = 'test_project_id'
    self._region = 'test_region'
    self._mock_api_client = mock.Mock()
    self._remote_dir = tempfile.mkdtemp()
    self._job_spec = _client._default_job_spec(self._region, self._image_uri, [
        '--remote_dir', self._remote_dir, '--distribution_strategy',
        MULTI_WORKER_MIRRORED_STRATEGY_NAME
    ])
    self._model = self._model()
    self._x = [[9.], [10.], [11.]] * 10
    self._y = [[xi[0] / 2. + 6] for xi in self._x]
    self._dataset = tf.data.Dataset.from_tensor_slices((self._x, self._y))
    self._scalar_fit_kwargs = {'batch_size': 1, 'epochs': 2, 'verbose': 3}

  def _set_up_training_mocks(self):
    self._mock_create = mock.Mock()
    self._mock_api_client.projects().jobs().create = self._mock_create
    self._mock_get = mock.Mock()
    self._mock_create.return_value = self._mock_get
    self._mock_get.execute.return_value = {
        'state': 'SUCCEEDED',
    }

  def _model(self):
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

  def test_default_job_spec(self):
    self.assertStartsWith(self._job_spec['job_id'], 'cloud_trainer_')
    self.assertDictContainsSubset(
        {
            'masterConfig': {
                'imageUri': self._image_uri,
            },
            'args': [
                '--remote_dir', self._remote_dir, '--distribution_strategy',
                MULTI_WORKER_MIRRORED_STRATEGY_NAME
            ],
        }, self._job_spec['trainingInput'])

  @mock.patch.object(discovery, 'build', autospec=True)
  def test_submit_job(self, mock_discovery_build):
    mock_discovery_build.return_value = self._mock_api_client
    self._set_up_training_mocks()

    _client._submit_job(self._job_spec, self._project_id)
    mock_discovery_build.assert_called_once_with(
        'ml', 'v1', cache_discovery=False)

    self._mock_create.assert_called_with(
        body=mock.ANY, parent='projects/{}'.format(self._project_id))

    _, fit_kwargs = list(self._mock_create.call_args)
    body = fit_kwargs['body']
    self.assertDictContainsSubset(
        {
            'masterConfig': {
                'imageUri': self._image_uri,
            },
            'args': [
                '--remote_dir', self._remote_dir, '--distribution_strategy',
                MULTI_WORKER_MIRRORED_STRATEGY_NAME
            ],
        }, body['trainingInput'])
    self.assertStartsWith(body['job_id'], 'cloud_trainer_')
    self._mock_get.execute.assert_called_with()

  def test_serialize_assets(self):
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=self._remote_dir)
    args = self._scalar_fit_kwargs
    args['callbacks'] = [tensorboard_callback]

    _client._serialize_assets(self._remote_dir, self._model, **args)
    self.assertGreaterEqual(
        len(
            tf.io.gfile.listdir(
                os.path.join(self._remote_dir, 'training_assets'))), 1)
    self.assertGreaterEqual(
        len(tf.io.gfile.listdir(os.path.join(self._remote_dir, 'model'))), 1)

    if tf.version.VERSION.split('.')[0] == '1':
      training_assets_graph = tf.compat.v2.saved_model.load(
          export_dir=os.path.join(self._remote_dir, 'training_assets'),
          tags=None)
    else:
      training_assets_graph = tf.saved_model.load(
          os.path.join(self._remote_dir, 'training_assets'))

    pickled_callbacks = tfds.as_numpy(training_assets_graph.callbacks_fn())
    unpickled_callbacks = cloudpickle.loads(pickled_callbacks)
    self.assertIsInstance(unpickled_callbacks[0],
                          tf.keras.callbacks.TensorBoard)

  @mock.patch.object(_client, '_submit_job', autospec=True)
  def test_fit_kwargs(self, mock_submit_job):
    job_id = _client.cloud_fit(
        self._model,
        x=self._dataset,
        validation_data=self._dataset,
        remote_dir=self._remote_dir,
        region=self._region,
        project_id=self._project_id,
        image_uri=self._image_uri,
        batch_size=1,
        epochs=2,
        verbose=3)

    kargs, _ = mock_submit_job.call_args
    body, _ = kargs
    self.assertEqual(body['job_id'], job_id)
    remote_dir = body['trainingInput']['args'][1]

    if tf.version.VERSION.split('.')[0] == '1':
      training_assets_graph = tf.compat.v2.saved_model.load(
          export_dir=os.path.join(remote_dir, 'training_assets'), tags=None)
    else:
      training_assets_graph = tf.saved_model.load(
          os.path.join(remote_dir, 'training_assets'))
    elements = training_assets_graph.fit_kwargs_fn()
    self.assertDictContainsSubset(
        tfds.as_numpy(elements), {
            'batch_size': 1,
            'epochs': 2,
            'verbose': 3
        })

  @mock.patch.object(_client, '_submit_job', autospec=True)
  def test_custom_job_spec(self, mock_submit_job):
    _client.cloud_fit(
        self._model,
        x=self._dataset,
        validation_data=self._dataset,
        remote_dir=self._remote_dir,
        job_spec=self._job_spec,
        batch_size=1,
        epochs=2,
        verbose=3)

    kargs, _ = mock_submit_job.call_args
    body, _ = kargs
    self.assertDictContainsSubset(
        {
            'masterConfig': {
                'imageUri': self._image_uri,
            },
            'args': [
                '--remote_dir', self._remote_dir, '--distribution_strategy',
                MULTI_WORKER_MIRRORED_STRATEGY_NAME
            ],
        }, body['trainingInput'])

  @mock.patch.object(_client, '_submit_job', autospec=True)
  @mock.patch.object(_client, '_serialize_assets', autospec=True)
  def test_distribution_strategy(self, mock_serialize_assets, mock_submit_job):

    _client.cloud_fit(self._model, x=self._dataset, remote_dir=self._remote_dir)

    kargs, _ = mock_submit_job.call_args
    body, _ = kargs
    self.assertDictContainsSubset(
        {
            'args': [
                '--remote_dir', self._remote_dir, '--distribution_strategy',
                MULTI_WORKER_MIRRORED_STRATEGY_NAME
            ],
        }, body['trainingInput'])

    _client.cloud_fit(
        self._model,
        x=self._dataset,
        remote_dir=self._remote_dir,
        distribution_strategy=MIRRORED_STRATEGY_NAME,
        job_spec=self._job_spec)

    kargs, _ = mock_submit_job.call_args
    body, _ = kargs
    self.assertDictContainsSubset(
        {
            'args': [
                '--remote_dir', self._remote_dir, '--distribution_strategy',
                MIRRORED_STRATEGY_NAME
            ],
        }, body['trainingInput'])

    with self.assertRaises(ValueError):
      _client.cloud_fit(
          self._model,
          x=self._dataset,
          remote_dir=self._remote_dir,
          distribution_strategy='not_implemented_strategy',
          job_spec=self._job_spec)


if __name__ == '__main__':
  tf.test.main()
