# Kvistur

Kvistur is a BiLSTM-based compound word analyzer. The work is described in the paper [Kvistur 2.0: a BiLSTM Compound Splitter for Icelandic](https://arxiv.org/abs/2004.07776).

# Train on GermaNet 12.0
```
wget http://www.sfs.uni-tuebingen.de/GermaNet/documents/compounds/split_compounds_from_GermaNet12.0.txt
python scripts/preprocess_germanet12.py split_compounds_from_GermaNet12.0.txt germanet.txt
python train.py --train germanet.txt --model-dir germanet
```

# Requirements
* Python 3.7+
* Tensorflow
* Scikit-learn

## License
    Copyright © 2020 Jón Friðrik Daðason

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.