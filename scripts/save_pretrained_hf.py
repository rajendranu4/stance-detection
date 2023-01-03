from pathlib import Path

import typer
from allennlp.common import util as common_util
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor

def main(archive_file: str, save_directory: Path) -> None:
    save_directory = Path(save_directory)
    save_directory.parents[0].mkdir(parents=True, exist_ok=True)

    common_util.import_module_and_submodules("StD")
    # cuda_device -1 places the model onto the CPU before saving. This avoids issues with
    # distributed models.
    overrides = "{'trainer.cuda_device': -1}"
    archive = load_archive(archive_file, overrides=overrides)
    predictor = Predictor.from_archive(archive, predictor_name="StD")

    token_embedder = predictor._model._text_field_embedder._token_embedders["tokens"]
    model = token_embedder.transformer_model
    tokenizer = token_embedder.tokenizer

    model.save_pretrained(str(save_directory))
    tokenizer.save_pretrained(str(save_directory))


if __name__ == "__main__":
    typer.run(main)
