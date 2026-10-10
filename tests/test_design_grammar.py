import unittest

from app.engine.grammar import infer_design_grammar


class DesignGrammarTests(unittest.TestCase):
    def test_empty_input_reports_no_measurable_contours(self):
        result = infer_design_grammar([])
        self.assertEqual(result["contour_count"], 0)
        self.assertEqual(result["method"], "opencv_contour_heuristics")
        self.assertTrue(result["rules"])

    def test_repeated_square_contours_are_counted(self):
        square_a = [[0, 0], [20, 0], [20, 20], [0, 20]]
        square_b = [[30, 0], [50, 0], [50, 20], [30, 20]]
        result = infer_design_grammar([square_a, square_b])
        self.assertEqual(result["contour_count"], 2)
        self.assertEqual(sum(result["primitive_counts"].values()), 2)
        self.assertTrue(any("Repeated motif family" in rule for rule in result["rules"]))


if __name__ == "__main__":
    unittest.main()
