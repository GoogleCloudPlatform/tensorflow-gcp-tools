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
"""Setup configuration of TensorFlow Enterprise client-side library."""


def make_required_install_packages():
  return [
      'absl-py',
      'cloudpickle',
      'google-api-python-client',
      'google-auth',
      'keras-tuner',
      'tensorflow>=1.15.0,<3.0',
      'tensorflow_datasets<3.1.0',
  ]


def make_required_test_packages():
  return ['mock', 'numpy']
