import unittest

import numpy as np

from app.engine.vision import build_geometry_graph_from_contours


class VisionGraphTests(unittest.TestCase):
    def test_closed_contour_preserves_closing_edge(self):
        contour = np.array([[0, 0], [10, 0], [10, 10]], dtype=np.int32)
        graph = build_geometry_graph_from_contours([contour])
        self.assertEqual(graph.graph.number_of_nodes(), 3)
        self.assertEqual(graph.graph.number_of_edges(), 3)
        self.assertEqual(graph.connected_components(), 1)

    def test_empty_contours_return_empty_graph(self):
        graph = build_geometry_graph_from_contours([])
        self.assertEqual(graph.graph.number_of_nodes(), 0)
        self.assertEqual(graph.graph.number_of_edges(), 0)


if __name__ == "__main__":
    unittest.main()
