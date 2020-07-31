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
"""End to end integration test for cloud_fit."""

import os
import time
from typing import Text
import uuid
from googleapiclient import discovery
import numpy as np
import tensorflow as tf
from tensorflow_enterprise_addons.cloud_fit import client
from tensorflow_enterprise_addons.cloud_fit import utils

# Can only export Datasets which were created executing eagerly
utils.enable_eager_for_tf_1()

MIRRORED_STRATEGY_NAME = utils.MIRRORED_STRATEGY_NAME
MULTI_WORKER_MIRRORED_STRATEGY_NAME = utils.MULTI_WORKER_MIRRORED_STRATEGY_NAME

# The staging bucket to use to copy the model and data for remote run.
REMOTE_DIR = os.environ['TEST_BUCKET']

# The project id to use to run tests.
PROJECT_ID = os.environ['PROJECT_ID']

# The GCP region in which the end-to-end test is run.
REGION = os.environ['REGION']

# The base docker image to use for remote environment.
DOCKER_IMAGE = os.environ['DOCKER_IMAGE']

# Using the build ID for testing
BUILD_ID = os.environ['BUILD_ID']

# Polling interval for AI Platform job status check
POLLING_INTERVAL_IN_SECONDS = 120


class CloudFitIntegrationTest(tf.test.TestCase):

  def setUp(self):
    super(CloudFitIntegrationTest, self).setUp()
    self.image_uri = DOCKER_IMAGE
    self.project_id = PROJECT_ID
    self.remote_dir = REMOTE_DIR
    self.region = REGION
    self.test_folders = []

  def tearDown(self):
    super(CloudFitIntegrationTest, self).tearDown()
    for folder in self.test_folders:
      self.delete_dir(folder)

  def delete_dir(self, path):
    """Deletes a directory if exists."""
    if tf.io.gfile.isdir(path):
      tf.io.gfile.rmtree(path)

  def model(self):
    inputs = tf.keras.layers.Input(shape=(3,))
    outputs = tf.keras.layers.Dense(2)(inputs)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='Adam', loss='mse', metrics=['mae'])
    return model

  def test_in_memory_data(self):
    # Create a folder under remote dir for this test's data
    tmp_folder = str(uuid.uuid4())
    remote_dir = os.path.join(self.remote_dir, tmp_folder)

    # Keep track of test folders created for final clean up
    self.test_folders.append(remote_dir)

    x = np.random.random((2, 3))
    y = np.random.randint(0, 2, (2, 2))

    # TF 1.x is not supported
    if utils.is_tf_v1():
      with self.assertRaises(RuntimeError):
        client.cloud_fit(
            self.model(),
            x=x,
            y=y,
            remote_dir=remote_dir,
            region=self.region,
            project_id=self.project_id,
            image_uri=self.image_uri,
            epochs=2)
      return

    job_id = client.cloud_fit(
        self.model(),
        x=x,
        y=y,
        remote_dir=remote_dir,
        region=self.region,
        project_id=self.project_id,
        image_uri=self.image_uri,
        job_id='cloud_fit_e2e_test_{}_{}'.format(
            BUILD_ID.replace('-', '_'), 'test_in_memory_data'),
        epochs=2)

    # Wait for AIP Training job to finish
    job_name = 'projects/{}/jobs/{}'.format(self.project_id, job_id)

    # Configure AI Platform training job
    api_client = discovery.build('ml', 'v1')
    request = api_client.projects().jobs().get(name=job_name)
    response = request.execute()
    while response['state'] not in ('SUCCEEDED', 'FAILED'):
      time.sleep(POLLING_INTERVAL_IN_SECONDS)
      response = request.execute()
    self.assertEqual(response['state'], 'SUCCEEDED')


if __name__ == '__main__':
  tf.test.main()
