import unittest

from glls.cli.visualize import GlobalSystem, _runtime_weight_config


class RuntimeWeightConfigChecks(unittest.TestCase):
    def test_abound_visa_uses_visa_adapted_files(self):
        cfg = _runtime_weight_config("visa", "ABounD", "1", "/unused/adaptclip")

        self.assertEqual(cfg["dataset_name"], "visa")
        self.assertEqual(cfg["localizer_name"], "abound")
        self.assertTrue(cfg["checkpoint_path"].endswith("/glls-abound-1shot/model"))
        self.assertTrue(cfg["save_path"].endswith("/glls-abound-1shot"))
        self.assertTrue(cfg["dataset_weight_paths"]["lora"].endswith("/visa/final_vvclip_model_state_visa.pth"))
        self.assertTrue(cfg["dataset_weight_paths"]["soft_prompt"].endswith("/visa/final_soft_prompt_state_visa.pth"))
        self.assertTrue(cfg["dataset_weight_paths"]["memory_bank"].endswith("/visa/final_memory_bank_visa.pt"))

    def test_abound_signature_changes_between_mvtec_and_visa(self):
        mvtec_cfg = _runtime_weight_config("mvtec", "ABounD", "1", "/unused/adaptclip")
        visa_cfg = _runtime_weight_config("visa", "ABounD", "1", "/unused/adaptclip")

        self.assertNotEqual(mvtec_cfg["signature"], visa_cfg["signature"])
        self.assertIn("final_memory_bank_mvtec.pt", mvtec_cfg["signature"])
        self.assertIn("final_memory_bank_visa.pt", visa_cfg["signature"])

    def test_adaptclip_visa_resolves_visa_checkpoint(self):
        cfg = _runtime_weight_config(
            "visa",
            "AdaptCLIP",
            "0",
            "/home/runzhi/data/kragad/models/AdaptCLIP",
        )

        self.assertEqual(cfg["localizer_name"], "adaptclip")
        self.assertTrue(cfg["checkpoint_path"].endswith("/checkpoints/visa_epoch_15.pth"))
        self.assertIn("visa_epoch_15.pth", cfg["signature"])

    def test_auto_mode_selects_abound_for_one_shot_and_adaptclip_for_zero_shot(self):
        one_shot = _runtime_weight_config("visa", "Auto", "1", "/unused/adaptclip")
        zero_shot = _runtime_weight_config("visa", "Auto", "0", "/home/runzhi/data/kragad/models/AdaptCLIP")

        self.assertEqual(one_shot["localizer_name"], "abound")
        self.assertEqual(zero_shot["localizer_name"], "adaptclip")

    def test_runtime_match_blocks_stale_dataset_weights(self):
        system = GlobalSystem()
        loaded = _runtime_weight_config("mvtec", "ABounD", "1", "/unused/adaptclip")
        system.is_initialized = True
        system.current_runtime_signature = loaded["signature"]
        system.current_weight_config = loaded

        ok, selected, message = system.runtime_match_status("visa", "ABounD", "1", "/unused/adaptclip")

        self.assertFalse(ok)
        self.assertEqual(selected["dataset_name"], "visa")
        self.assertIn("final_memory_bank_mvtec.pt", message)
        self.assertIn("final_memory_bank_visa.pt", message)


if __name__ == "__main__":
    unittest.main()
