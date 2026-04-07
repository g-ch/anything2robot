import argparse
import importlib
import pickle


class _CompatObject:
    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self._state = state


class RobotOptResult(_CompatObject):
    pass


class LinkResult(_CompatObject):
    pass


class TreeNode(_CompatObject):
    pass


class Link(_CompatObject):
    pass


class Line(_CompatObject):
    pass


class MeshGroup(_CompatObject):
    pass


_CLASS_MAP = {
    ("modules.interference_removal", "RobotOptResult"): RobotOptResult,
    ("modules.interference_removal", "LinkResult"): LinkResult,
    ("data_struct", "TreeNode"): TreeNode,
    ("mesh_loader", "Link"): Link,
    ("mesh_loader", "Line"): Line,
    ("modules.mesh_loader", "Link"): Link,
    ("modules.mesh_loader", "Line"): Line,
    ("modules.mesh_decomp", "Mesh_Group"): MeshGroup,
}


class CompatRobotResultUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        key = (module, name)
        if key in _CLASS_MAP:
            return _CLASS_MAP[key]

        if module == "argparse" and name == "Namespace":
            return argparse.Namespace

        if module.startswith("numpy") or module == "builtins":
            imported = importlib.import_module(module)
            return getattr(imported, name)

        return super().find_class(module, name)


def load_robot_result(path):
    with open(path, "rb") as handle:
        return CompatRobotResultUnpickler(handle).load()
