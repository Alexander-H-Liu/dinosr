
# [DinoSR: Self-Distillation and Online Clustering for Self-supervised Speech Representation Learning](https://arxiv.org/pdf/2305.10005.pdf)


### Setup

- Codebase preparation (based on [`fairseq`](https://github.com/facebookresearch/fairseq))
```
# we use fairseq to build the model
git clone https://github.com/facebookresearch/fairseq
cd fairseq
git checkout 47e279842ac8776e3964b0e45c320ad1d2ea6096  # we recommend using the commit DinoSR was developed on
pip install --editable ./

# plug in DinoSR
cd examples
git clone https://github.com/Alexander-H-Liu/dinosr.git
```

- Data preparation:
please follow [`instruction provided by wav2vec2`](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec) for pre-training/fine-tuning data preprocessing


### Usage

- Training

    For the list of hyper-parameters, see [`config file`](config/base.yaml) and also [`model attributes`](models/dinosr.py) where default settings used in the paper are provided. 

```
# minimal example to reproduce model
python fairseq_cli/hydra_train.py -m \
    --config-dir examples/dinosr/config/ \
    --config-name base \
    task.data=/path/to/prepared/librispeech/ \
    common.user_dir=examples/dinosr &
```

- Loading pre-trained model as python object

```
import fairseq
import argparse
code_path = "examples/dinosr"
fairseq.utils.import_user_module(argparse.Namespace(user_dir=code_path))
ckpt_path = "/path/to/the/checkpoint.pt"
models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
model = models[0]
```

- Fine-tuning pre-trained checkpoint as ASR

```
# minimal example for fine-tuning with 100hr data
python fairseq_cli/hydra_train.py -m \
        --config-dir examples/wav2vec/config/finetuning \
        --config-name base_100h \
        common.user_dir=examples/dinosr \
        task.data=/path/to/labeled/librispeech/ \
        model.w2v_path=/path/to/dinosr.ckpt \
        task.normalize=True
```

### Pre-trained checkpoint

Pre-trained checkpoint without fine-tuning can be downloaded [here](https://data.csail.mit.edu/placesaudio/dinosr/dinosr.ckpt).
