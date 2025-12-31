from pathlib import Path

def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 30,
        "lr": 1.0,
        "seq_len": 350,
        "d_model": 512,
        "datasource": 'Helsinki-NLP/opus_books',
        "lang_src": "en",
        "lang_tgt": "it",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": None,
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel",
        "save_best_only": False,
        "save_every": None
    }

def get_weights_file_path(config, epoch: str):
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)