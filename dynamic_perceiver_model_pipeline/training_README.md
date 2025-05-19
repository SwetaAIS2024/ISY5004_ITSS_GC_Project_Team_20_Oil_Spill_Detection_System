dynamic_perceiver_model_pipeline/
├── cnn_perceiver_module.py      # Dynamic Perceiver classifier
├── cnn_pretrain_module.py       # mobilenet_v3_small 
├── training_module.py       # Training logic for Perceiver
├── eval.py                  # Accuracy evaluation on datasets
├── main.py                  # Full pipeline: detect, train, evaluate, infer
├── data/                    # Directory for training and evaluation data
│   ├── original/
│   ├── synthetic/
│   └── combined/
└── README.md                # This file

#dataset folder organisation:
data/
├── original/
│   ├── 0/  ← non-spill
│   └── 1/  ← oil spill
├── synthetic/
│   ├── 0/
│   └── 1/
├── combined/
│   ├── 0/
│   └── 1/

