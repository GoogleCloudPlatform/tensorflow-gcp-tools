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
"""Setup configuration of TensorFlow Enterprise client-side library."""

import importlib
import types
import setuptools

import dependencies

# It should not do `from tensorflow_enterprise_addons import version`,
# because setuptools is executed before dependencies may be installed,
# hence __init__.py may fail.
loader = importlib.machinery.SourceFileLoader(
    fullname='version',
    path='tensorflow_enterprise_addons/version.py',
)
version = types.ModuleType(loader.name)
loader.exec_module(version)


setuptools.setup(
    name='tensorflow-enterprise-addons',
    version=version.__version__,
    license='Apache 2.0',
    description=(
        'Client-side library suites of TensorFlow Enteprise on Google Cloud '
        'Platform (GCP), which implements a special integration with GCP '
        'behind the TensorFlow APIs.'
    ),
    install_requires=dependencies.make_required_install_packages(),
    extras_require={
        'test': dependencies.make_required_test_packages()
    },
    packages=setuptools.find_packages(),
    package_data={'tensorflow_enterprise_addons': ['cloudtuner/api/*.json']},
    include_package_data=True,
    author='Google LLC',
)
