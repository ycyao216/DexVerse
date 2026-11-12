"""Requires Isaac Lab Python (pxr); no simulator process is started."""
import sys
import tempfile
import unittest
from pathlib import Path

from pxr import Usd, UsdGeom, UsdPhysics

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "asset_tools"))
from download_selected_objects import _patch_usd


class CollisionPreparationTest(unittest.TestCase):
    def test_authors_colliders_on_visible_meshes_and_is_repeatable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "object.usda"
            stage = Usd.Stage.CreateNew(str(path))
            mesh = UsdGeom.Mesh.Define(stage, "/Object/mesh")
            hidden = UsdGeom.Mesh.Define(stage, "/Object/hidden")
            hidden.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
            stage.GetRootLayer().Save()
            del stage
            self.assertEqual(_patch_usd(path, "convexDecomposition"), 1)
            self.assertEqual(_patch_usd(path, "convexDecomposition"), 1)
            stage = Usd.Stage.Open(str(path))
            visible = stage.GetPrimAtPath("/Object/mesh")
            self.assertTrue(visible.HasAPI(UsdPhysics.CollisionAPI))
            self.assertTrue(UsdPhysics.CollisionAPI(visible).GetCollisionEnabledAttr().Get())
            self.assertEqual(UsdPhysics.MeshCollisionAPI(visible).GetApproximationAttr().Get(), "convexDecomposition")
            self.assertFalse(stage.GetPrimAtPath("/Object/hidden").HasAPI(UsdPhysics.CollisionAPI))


if __name__ == "__main__":
    unittest.main()
