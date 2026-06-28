import unittest

from glls.runtime_config import (
    default_adaptclip_checkpoint,
    resolve_localizer_name,
    runtime_weight_config,
)


class RuntimeConfigTest(unittest.TestCase):
    def test_auto_uses_abound_only_for_mvtec_and_visa_one_shot(self):
        self.assertEqual(resolve_localizer_name("auto", "mvtec", 1), "abound")
        self.assertEqual(resolve_localizer_name("Auto", "VisA", "1"), "abound")

        self.assertEqual(resolve_localizer_name("auto", "mvtec", 0), "adaptclip")
        self.assertEqual(resolve_localizer_name("auto", "visa", 0), "adaptclip")
        self.assertEqual(resolve_localizer_name("auto", "mpdd", 1), "adaptclip")
        self.assertEqual(resolve_localizer_name("auto", "dagm", 0), "adaptclip")

    def test_abound_runtime_config_uses_dataset_specific_release_artifacts(self):
        cfg = runtime_weight_config(
            "visa",
            "auto",
            1,
            adaptclip_ckpt_path="/unused/adaptclip/checkpoint.pth",
            abound_model_path="/models/glls-abound-1shot/model",
            abound_save_path="/models/glls-abound-1shot",
        )

        self.assertEqual(cfg["localizer_name"], "abound")
        self.assertEqual(cfg["image_size"], 336)
        self.assertEqual(cfg["checkpoint_path"], "/models/glls-abound-1shot/model")
        self.assertEqual(cfg["save_path"], "/models/glls-abound-1shot")
        self.assertTrue(cfg["dataset_weight_paths"]["lora"].endswith("/visa/final_vvclip_model_state_visa.pth"))
        self.assertTrue(cfg["dataset_weight_paths"]["soft_prompt"].endswith("/visa/final_soft_prompt_state_visa.pth"))
        self.assertTrue(cfg["dataset_weight_paths"]["memory_bank"].endswith("/visa/final_memory_bank_visa.pt"))

    def test_adaptclip_runtime_config_keeps_shot_specific_threshold_route(self):
        cfg = runtime_weight_config(
            "mvtec",
            "auto",
            0,
            adaptclip_ckpt_path="/models/AdaptCLIP/checkpoints/mvtec_epoch_15.pth",
        )

        self.assertEqual(cfg["localizer_name"], "adaptclip")
        self.assertEqual(cfg["k_shot"], 0)
        self.assertEqual(cfg["image_size"], 518)
        self.assertEqual(cfg["checkpoint_path"], "/models/AdaptCLIP/checkpoints/mvtec_epoch_15.pth")
        self.assertEqual(cfg["dataset_weight_paths"], {"checkpoint": "/models/AdaptCLIP/checkpoints/mvtec_epoch_15.pth"})

    def test_default_adaptclip_checkpoint_uses_visa_domain_only_for_visa(self):
        self.assertTrue(default_adaptclip_checkpoint("visa", "/models/AdaptCLIP").endswith("/checkpoints/visa_epoch_15.pth"))
        self.assertTrue(default_adaptclip_checkpoint("mvtec", "/models/AdaptCLIP").endswith("/checkpoints/mvtec_epoch_15.pth"))
        self.assertTrue(default_adaptclip_checkpoint("mpdd", "/models/AdaptCLIP").endswith("/checkpoints/mvtec_epoch_15.pth"))


if __name__ == "__main__":
    unittest.main()
