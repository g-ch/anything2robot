import sys
from PyQt5 import QtWidgets

class TreeWidgetExample(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        # Set up the layout
        layout = QtWidgets.QVBoxLayout()

        # Create a QTreeWidget
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabel("Tree Structure Example")

        # Add root item
        root_item = QtWidgets.QTreeWidgetItem(self.tree, ["Root"])
        
        # Add first level children under root
        child_item_1 = QtWidgets.QTreeWidgetItem(root_item, ["Child 1"])
        child_item_2 = QtWidgets.QTreeWidgetItem(root_item, ["Child 2"])

        # Add second level children under Child 1
        sub_child_item_1 = QtWidgets.QTreeWidgetItem(child_item_1, ["Sub-child 1"])
        sub_child_item_2 = QtWidgets.QTreeWidgetItem(child_item_1, ["Sub-child 2"])

        # Add second level children under Child 2
        sub_child_item_3 = QtWidgets.QTreeWidgetItem(child_item_2, ["Sub-child 3"])
        sub_child_item_4 = QtWidgets.QTreeWidgetItem(child_item_2, ["Sub-child 4"])

        # Expand all nodes by default
        self.tree.expandAll()

        # Add the tree widget to the layout
        layout.addWidget(self.tree)
        
        # Set the layout for the main window
        self.setLayout(layout)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = TreeWidgetExample()
    window.show()
    sys.exit(app.exec_())



# import sys
# import pyvista as pv
# from PyQt5 import QtWidgets
# from pyvistaqt import QtInteractor


# class MainWindow(QtWidgets.QMainWindow):
#     def __init__(self, parent=None):
#         super(MainWindow, self).__init__(parent)

#         # Set up the main window
#         self.setWindowTitle("PyVista in Qt Window")
#         self.setGeometry(300, 100, 800, 600)

#         # Create a central widget and layout
#         self.frame = QtWidgets.QFrame()
#         self.layout = QtWidgets.QVBoxLayout()

#         # Create a PyVista plotter within the Qt window
#         self.plotter = QtInteractor(self.frame)
#         self.layout.addWidget(self.plotter.interactor)

#         # Set the layout to the central widget
#         self.frame.setLayout(self.layout)
#         self.setCentralWidget(self.frame)

#         # Load an STL file and add a sphere to the PyVista plotter
#         self.load_model()

#     def load_model(self):
#         # Load STL file
#         stl_file_path = "/media/clarence/Clarence/anything2robot/result/gold_lynel_20241018-170744/result_round3/scaled_model_expected_x_60.500000000000014.stl"

#         stl_mesh = pv.read(stl_file_path)

#         # Create a sphere (representing a point)
#         sphere = pv.Sphere(radius=0.1, center=(1, 1, 1))

#         # Add the STL mesh with transparency
#         self.plotter.add_mesh(stl_mesh, opacity=0.5, color="lightblue")

#         # Add a red sphere
#         self.plotter.add_mesh(sphere, color="red")

#         # Show the plot
#         self.plotter.show()


# if __name__ == "__main__":
#     app = QtWidgets.QApplication(sys.argv)
#     window = MainWindow()
#     window.show()
#     sys.exit(app.exec_())


# # stl_file_path = "/media/clarence/Clarence/anything2robot/result/gold_lynel_20241018-170744/result_round3/scaled_model_expected_x_60.500000000000014.stl"
