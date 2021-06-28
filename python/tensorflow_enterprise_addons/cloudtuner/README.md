
# CloudTuner

## What is this module?

`cloudtuner` is a module which is part of the broader
`tensorflow_enterprise_addons`. This module is an implementation of a library
for hyperparameter tuning that is built into the
[KerasTuner](https://github.com/keras-team/keras-tuner) and creates a seamless
integration with
[Cloud AI Platform Optimizer Beta](https://cloud.google.com/ai-platform/optimizer/docs)
as a backend to get suggestions of hyperparameters and run trials.

## Installation

### Requirements

-   Python >= 3.5
-   Tensorflow >= 2.0
-   [Set up your Google Cloud project](https://cloud.google.com/ai-platform/docs/getting-started-keras#set_up_your_project)
-   [Authenticate your GCP account](https://cloud.google.com/ai-platform/docs/getting-started-keras#authenticate_your_gcp_account)
-   Enable [Google AI platform](https://cloud.google.com/ai-platform/) APIs on
    your GCP project.

For detailed end to end setup instructions, please see
[Setup instructions](#setup-instructions) section.

### Install latest release

```shell
pip install tensorflow-enterprise-addons
```

### Install from source

`cloudtuner` can be built and installed directly from source.

```shell
git clone https://github.com/GoogleCloudPlatform/tensorflow-gcp-tools.git
cd tensorflow-gcp-tools/python && python3 setup.py -q bdist_wheel
pip3 install -U dist/tensorflow_enterprise_addons-*.whl --quiet
```

## High level overview

`cloudtuner` module creates a seamless integration with
[Cloud AI Platform Optimizer Beta](https://cloud.google.com/ai-platform/optimizer/docs)
as a backend to get suggestions of hyperparameters and run trials.

```python
from tensorflow_enterprise_addons import cloudtuner
import keras_tuner
import tensorflow as tf

(x, y), (val_x, val_y) = tf.keras.datasets.mnist.load_data()
x = x.astype('float32') / 255.
val_x = val_x.astype('float32') / 255.

def build_model(hp):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    for _ in range(hp.get('num_layers')):
        model.add(tf.keras.layers.Dense(units=64, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=hp.get('learning_rate')),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    return model

# Configure the search space
HPS = keras_tuner.engine.hyperparameters.HyperParameters()
HPS.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
HPS.Int('num_layers', 2, 10)

# Instantiate CloudTuner
tuner = cloudtuner.CloudTuner(
    build_model,
    project_id=PROJECT_ID,
    region=REGION,
    objective='accuracy',
    hyperparameters=HPS,
    max_trials=5,
    directory='tmp_dir/1')

# Execute our search for the optimization study
tuner.search(x=x, y=y, epochs=10, validation_data=(val_x, val_y))

# Get a summary of the trials from this optimization study
tuner.results_summary()
```

## Setup instructions

End to end instructions to help set up your environment for `cloudtuner`. If you
are using
[Google Cloud Hosted Notebooks](https://cloud.google.com/ai-platform-notebooks)
you can skip the setup and authentication steps and start from step 8.

1.  Create a new local directory

    ```shell
    mkdir cloudtuner
    cd cloudtuner
    ```

1.  Make sure you have `python >= 3.5`

    ```shell
    python -V
    ```

1.  Setup virtual environment

    ```shell
    virtualenv venv --python=python3
    source venv/bin/activate
    ```

1.  [Set up your Google Cloud project](https://cloud.google.com/ai-platform/docs/getting-started-keras#set_up_your_project)

    Verify that gcloud sdk is installed.

    ```shell
    which gcloud
    ```

    Set default gcloud project

    ```shell
    export PROJECT_ID=<your-project-id>
    gcloud config set project $PROJECT_ID
    ```

1.  [Authenticate your GCP account](https://cloud.google.com/ai-platform/docs/getting-started-keras#authenticate_your_gcp_account)
    Follow the prompts after executing the command. This is not required in
    hosted notebooks.

    ```shell
    gcloud auth application-default
    ```

1.  Authenticate Docker to access Google Cloud Container Registry (gcr)

    To use local [Docker] with google
    [Cloud Container Registry](https://cloud.google.com/container-registry/docs/advanced-authentication)
    for docker build and publish

    ```shell
    gcloud auth configure-docker
    ```

1.  Create a Bucket for Data and Model transfer
    [Create a cloud storage bucket](https://cloud.google.com/ai-platform/docs/getting-started-keras#create_a_bucket)
    for `cloudtuner` temporary storage.

    ```shell
    export BUCKET_NAME="your-bucket-name"
    export REGION="us-central1"
    gsutil mb -l $REGION gs://$BUCKET_NAME
    ```

1.  Build and install latest release

    ```shell
    git clone https://github.com/GoogleCloudPlatform/tensorflow-gcp-tools.git
    cd tensorflow-gcp-tools/python && python3 setup.py -q bdist_wheel
    pip3 install -U dist/tensorflow_enterprise_addons-*.whl --quiet
    ```

You are all set! You can now follow our
[examples](https://github.com/GoogleCloudPlatform/tensorflow-gcp-tools/blob/master/examples/ai_platform_optimizer_tuner.ipynb)
to try out `cloudtuner`.

## License

[Apache License 2.0](https://github.com/GoogleCloudPlatform/tensorflow-gcp-tools/blob/master/LICENSE)
