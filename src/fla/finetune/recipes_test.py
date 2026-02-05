import fla.training.config as training_config
from fla.finetune import build_train_config
from fla.finetune import list_recipes
from fla.finetune import RecipeOverrides


def _base_overrides():
    return RecipeOverrides(
        repo_ids=("lerobot/test_dataset",),
        repo_id_to_prompt={"lerobot/test_dataset": "Pick up the block"},
        exp_name="unit_test",
        action_dim=7,
    )


def test_list_recipes_nonempty():
    recipes = list_recipes()
    assert recipes
    assert any(r.name == "pi0_frozen_backbone" for r in recipes)
    assert any(r.name == "pi05_frozen_backbone" for r in recipes)


def test_build_train_config_basic():
    config = build_train_config("pi0_frozen_backbone", _base_overrides())
    assert isinstance(config, training_config.TrainConfig)
    assert config.data.repo_ids == ("lerobot/test_dataset",)
    assert config.model.action_dim == 7


def test_build_train_config_lora():
    config = build_train_config("pi0_lora", _base_overrides())
    assert "lora" in config.model.paligemma_variant
    assert "lora" in config.model.action_expert_variant
