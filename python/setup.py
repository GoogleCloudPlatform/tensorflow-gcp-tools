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

with open('README.md') as f:
  long_description = f.read()

setuptools.setup(
    author='Google LLC',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    description=(
        'Client-side library suite of TensorFlow Enterprise on Google Cloud'
        'Platform (GCP), which implements specific integration between GCP and'
        'TensorFlow APIs.'),
    extras_require={'tests': dependencies.make_required_test_packages()},
    include_package_data=True,
    install_requires=dependencies.make_required_install_packages(),
    keywords='tensorflow enterprise gcp google cloud addons cloudAI deep learning hyperparameter tuning',
    license='Apache 2.0',
    name='tensorflow-enterprise-addons',
    package_data={'tensorflow_enterprise_addons': ['cloudtuner/api/*.json']},
    packages=setuptools.find_packages(),
    project_urls={
        'Source': 'https://github.com/GoogleCloudPlatform/tensorflow-gcp-tools',
    },
    python_requires='>=3.5, <4',
    url='https://github.com/GoogleCloudPlatform/tensorflow-gcp-tools',
    version=version.__version__,
)
