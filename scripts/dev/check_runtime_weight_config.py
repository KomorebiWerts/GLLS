import unittest

from glls.runtime_config import runtime_weight_config


class RuntimeWeightConfigChecks(unittest.TestCase):
    def test_abound_visa_uses_visa_release_files(self):
        cfg = runtime_weight_config(
            "visa",
            "Auto",
            "1",
            "/unused/adaptclip",
            abound_model_path="/models/glls-abound-1shot/model",
            abound_save_path="/models/glls-abound-1shot",
        )

        self.assertEqual(cfg["dataset_name"], "visa")
        self.assertEqual(cfg["localizer_name"], "abound")
        self.assertTrue(cfg["checkpoint_path"].endswith("/glls-abound-1shot/model"))
        self.assertTrue(cfg["save_path"].endswith("/glls-abound-1shot"))
        self.assertTrue(cfg["dataset_weight_paths"]["lora"].endswith("/visa/final_vvclip_model_state_visa.pth"))
        self.assertTrue(cfg["dataset_weight_paths"]["soft_prompt"].endswith("/visa/final_soft_prompt_state_visa.pth"))
        self.assertTrue(cfg["dataset_weight_paths"]["memory_bank"].endswith("/visa/final_memory_bank_visa.pt"))

    def test_abound_signature_changes_between_mvtec_and_visa(self):
        mvtec_cfg = runtime_weight_config(
            "mvtec",
            "Auto",
            "1",
            "/unused/adaptclip",
            abound_model_path="/models/glls-abound-1shot/model",
            abound_save_path="/models/glls-abound-1shot",
        )
        visa_cfg = runtime_weight_config(
            "visa",
            "Auto",
            "1",
            "/unused/adaptclip",
            abound_model_path="/models/glls-abound-1shot/model",
            abound_save_path="/models/glls-abound-1shot",
        )

        self.assertNotEqual(mvtec_cfg["signature"], visa_cfg["signature"])
        self.assertIn("final_memory_bank_mvtec.pt", mvtec_cfg["signature"])
        self.assertIn("final_memory_bank_visa.pt", visa_cfg["signature"])

    def test_adaptclip_visa_zero_shot_resolves_visa_checkpoint(self):
        cfg = runtime_weight_config(
            "visa",
            "Auto",
            "0",
            "",
            adaptclip_root="/models/AdaptCLIP",
        )

        self.assertEqual(cfg["localizer_name"], "adaptclip")
        self.assertEqual(cfg["k_shot"], 0)
        self.assertTrue(cfg["checkpoint_path"].endswith("/checkpoints/visa_epoch_15.pth"))
        self.assertIn("visa_epoch_15.pth", cfg["signature"])

    def test_binary_ad_datasets_stay_on_adaptclip(self):
        for dataset in ("mpdd", "dtd", "dagm"):
            with self.subTest(dataset=dataset):
                cfg = runtime_weight_config(dataset, "Auto", "1", "/models/AdaptCLIP")
                self.assertEqual(cfg["localizer_name"], "adaptclip")
                self.assertEqual(cfg["image_size"], 518)


if __name__ == "__main__":
    unittest.main()
