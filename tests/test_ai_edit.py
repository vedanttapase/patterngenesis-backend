import unittest

from app.engine.ai_edit import apply_operation, parse_instruction


class ParseInstructionTests(unittest.TestCase):
    def test_set_ring_count_uses_target_not_delta(self):
        result = parse_instruction("set rings to 5", {"rings": 3, "nfold": 8})
        self.assertEqual(result["changes"], {"rings": 5})
        self.assertEqual(apply_operation({"rings": 3}, result)["rings"], 5)

    def test_decrease_rings_decreases(self):
        result = parse_instruction("decrease rings by 2", {"rings": 5, "nfold": 8})
        self.assertEqual(result["changes"]["rings"], 3)

    def test_increase_rings_increases(self):
        result = parse_instruction("increase rings by 2", {"rings": 3, "nfold": 8})
        self.assertEqual(result["changes"]["rings"], 5)

    def test_unsupported_3d_height_is_not_reported_as_applied(self):
        result = parse_instruction("make it taller by 10", {"rows": 7, "cols": 7})
        self.assertEqual(result["op"], "unknown")
        self.assertEqual(result["changes"], {})

    def test_grid_symmetry_rejects_unsupported_order(self):
        result = parse_instruction("make it 6-fold symmetry", {"rows": 7, "cols": 7, "symmetry": "4fold"})
        self.assertEqual(result["op"], "unknown")
        self.assertEqual(result["changes"], {})

    def test_regenerate_changes_radial_pattern_parameters(self):
        result = parse_instruction("try another version", {"nfold": 8, "rings": 3, "motif": "polygon", "motif_sides": 6})
        self.assertEqual(result["op"], "regenerate_seed")
        self.assertEqual(result["changes"], {"motif_sides": 7})

    def test_blank_instruction_is_safe(self):
        result = parse_instruction("   ", {"rows": 7, "cols": 7})
        self.assertEqual(result["op"], "unknown")
        self.assertEqual(result["changes"], {})


if __name__ == "__main__":
    unittest.main()
