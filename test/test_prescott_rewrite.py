import unittest
from pathlib import Path

import numpy as np

from python_model.prescott_rewrite import (
    PrescottPythonRewriter,
    canonical_edge01_config,
)


class PrescottRewriteParityTest(unittest.TestCase):
    def test_edge01_reference_outputs_match(self):
        reference_dir = Path(
            "Prescott_ephaptic_coupling_MRG_model/radius = 4/2 fibers_same diameter_aligned/edge dist = 0.1"
        )

        outputs = PrescottPythonRewriter(canonical_edge01_config()).run()

        checks = {
            "R.txt": outputs.R,
            "C.txt": outputs.C,
            "G.txt": outputs.G,
            "parameters.txt": outputs.parameters,
            "Number_of_nodes.txt": outputs.number_of_nodes,
            "unique_radius.txt": outputs.unique_radius,
            "neighboringAxon.txt": outputs.neighboring_axon,
            "dist.txt": outputs.dist,
            "Distance_edge.txt": outputs.distance_edge,
            "Boundary_coordinates.txt": outputs.boundary_coordinates,
            "Boundary_neighboring.txt": outputs.boundary_neighboring,
            "Boundary_dist.txt": outputs.boundary_dist,
            "Rg_11.txt": outputs.rg_11,
            "Boundary_Rg_1.txt": outputs.boundary_rg_1,
        }

        for filename, generated in checks.items():
            with self.subTest(filename=filename):
                reference = np.loadtxt(reference_dir / filename)
                generated = np.atleast_2d(generated)
                reference = np.atleast_2d(reference)
                self.assertEqual(reference.shape, generated.shape)
                self.assertTrue(np.allclose(reference, generated, rtol=1e-9, atol=1e-9))


if __name__ == "__main__":
    unittest.main()
