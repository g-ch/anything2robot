""" Mesh Loader Module

The mesh loader module is responsible for loading the mesh data and joint data from the file system 
and scale them to the appropriate size.

author: Moji Shi
date: 2024-03-01

"""
from plot_utils import *
from data_struct import *
import open3d as o3d
import ast
import os.path
import argparse
import tkinter as tk
import pickle as pkl
import threading
from threading import Thread
from tkinter import ttk, messagebox
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output
#import requests
from wsgiref.simple_server import make_server
from flask import Flask
import time
import pyvista as pv

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QDoubleSpinBox, QPushButton

import sys
import pyvista as pv
from pyvistaqt import QtInteractor


class Line:
    def __init__(self, start, end):
        self.start = np.array(start)
        self.end = np.array(end)

    def __eq__(self, __value: object) -> bool:
        return ((self.start == __value.start).all() and (self.end == __value.end).all()) or ((self.start == __value.end).all() and (self.end == __value.start).all())
    
    def get_distance(self, point):
        """
        Get the distance between the point and the line segment.
        """
        line_vec = self.end - self.start
        point_vec = point - self.start
        line_len_sq = np.dot(line_vec, line_vec)
        t = max(0, min(1, np.dot(point_vec, line_vec) / line_len_sq))
        closest = self.start + t * line_vec
        
        return np.linalg.norm(point - closest)
        
    
    def __str__(self):
        return f"Line: {self.start} -> {self.end}"

class Mesh:
    def __init__(self, mesh_path):
        """
        Load the mesh data from the file system.
        """
        self.mesh_path = mesh_path
        self.mesh_o3d = o3d.io.read_triangle_mesh(mesh_path)
        self.mesh_plotly = create_mesh(mesh_path)

    def transform(self, transformation_matrix):
        """
        Transform the mesh data.
        """
        self.mesh_o3d = self.mesh_o3d.transform(transformation_matrix)
        vertices = np.asarray(self.mesh_o3d.vertices)
        triangles = np.asarray(self.mesh_o3d.triangles)
        self.mesh_plotly = go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=triangles[:, 0],
            j=triangles[:, 1],
            k=triangles[:, 2],
            opacity=0.2,
            color='grey'
        )

    def scale(self, scale_factor):
        """
        Scale the mesh data.
        """
        
        self.mesh_o3d.scale(scale_factor, center=np.array([0, 0, 0]))
        
        vertices = np.asarray(self.mesh_o3d.vertices)
        triangles = np.asarray(self.mesh_o3d.triangles)
        self.mesh_plotly = go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=triangles[:, 0],
            j=triangles[:, 1],
            k=triangles[:, 2],
            opacity=0.2,
            color='grey'
        )

    
    def render(self, save_only=False, save_path=None):
        """
        Render the mesh data.
        """
        fig = go.Figure(data=[self.mesh_plotly])
        #fig.show()
        if not save_only:
            fig.show()
        if save_path is not None:
            fig.write_image(save_path)

class Link:
    def __init__(self, name):
        self.name = name
        self.joints = {}
        self.axis = None
        self.joint_lines = [] # list of Line

    def construct_joint_lines(self):
        """
        Construct the joint lines.
        """
        self.joint_lines = []
        for joint_name, joint_position in self.joints.items():
            for joint_name_2, joint_position_2 in self.joints.items():
                if joint_name != joint_name_2:
                    new_line = Line(joint_position, joint_position_2)
                    add_line = True
                    for line in self.joint_lines:
                        if line == new_line:
                            add_line = False
                            break
                    if add_line:
                        self.joint_lines.append(new_line)
    
    def add_joint(self, joint_name, joint_position):
        for origin_joints in self.joints.values():
            #TODO: Consider the case that the joint is already in the joint_lines and removing a line when a joint is removed
            self.joint_lines.append(Line(origin_joints, joint_position))
        self.joints[joint_name] = joint_position

    def add_joints(self, joint_dict):
        for joint_name, joint_position in joint_dict.items():
            self.add_joint(joint_name, joint_position)
    
    def add_axis(self, axis):
        if len(axis) == 6:
            self.axis = [axis[:3], axis[3:6]]
        elif len(axis) == 9:
            self.axis = [axis[:3], axis[3:6], axis[6:9]]

    def get_min_axis_distance(self, point):
        """
        Get the minimum distance between the point and the lines made by any two joints.
        """
        min_distance = float('inf')
        for line in self.joint_lines:
            distance = line.get_distance(point)
            if distance < min_distance:
                min_distance = distance
        return min_distance

    def __str__(self):
        return self.name
    

class LinkTreeGUI(QtWidgets.QMainWindow):
    def __init__(self, mesh, args):
        super().__init__()
        self.args = args
        self.mesh = mesh
        self.setWindowTitle("Link Tree Constructor")
        self.setGeometry(100, 100, 800, 600)
        
        # Layout configuration
        self.central_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.layout = QtWidgets.QGridLayout(self.central_widget)

        self.nodes = {}
        self.current_link = None

        self.start_design_flag = False
        
        # Tree view
        self.tree_view_frame = self.create_tree_view_frame()
        self.tree_view_frame.setMinimumSize(300, 300)  
        self.tree_view_frame.setMaximumSize(800, 800)  

        self.layout.addWidget(self.tree_view_frame, 0, 0)

        # Joints in current link
        self.joint_list_frame = self.create_joint_list_frame()
        self.joint_list_frame.setMinimumSize(300, 300) 
        self.joint_list_frame.setMaximumSize(800, 800)
        self.layout.addWidget(self.joint_list_frame, 0, 1)

        # Add link controls
        self.link_edit_frame = self.create_link_edit_frame()
        self.layout.addWidget(self.link_edit_frame, 1, 0)

        # Joint controls
        self.joint_edit_frame = self.create_joint_edit_frame()
        self.layout.addWidget(self.joint_edit_frame, 1, 1)

        # Axis controls
        self.axis_edit_frame = self.create_axis_edit_frame()
        self.layout.addWidget(self.axis_edit_frame, 2, 0)

        # Save controls
        self.save_frame = self.create_save_frame()
        self.layout.addWidget(self.save_frame, 2, 1)

        # Point selection frame
        self.point_selection_frame = self.create_point_selection_frame()
        self.point_selection_frame.setMinimumSize(400, 400) 
        self.point_selection_frame.setMaximumSize(800, 800)
        self.layout.addWidget(self.point_selection_frame, 0, 2, 2, 1)

        self.point_selection_slider_frame = self.create_point_selection_slider_frame()
        self.layout.addWidget(self.point_selection_slider_frame, 2, 2)

        # Plotly visualization in the Web UI using Dash
        self.fig = make_subplots(specs=[[{"type": "scene"}]])
        self.fig.add_trace(mesh.mesh_plotly)

        self.server = Flask(__name__)
        self.app = Dash(__name__, server=self.server)

        self.server = make_server("localhost", 8050, self.server)
        self.server_thread = threading.Thread(target=self.server.serve_forever)
        self.server_thread.start()
        self.app.layout = html.Div([
            html.H4('Interactive plot with custom data source'),
            dcc.Graph(id="graph", style={'width': '90vh', 'height': '90vh'}),
            html.Button("Update Data", id="update-button", n_clicks=0),
        ])

        @self.app.callback(
            Output("graph", "figure"), 
            Input("update-button", "n_clicks"))
        def update_bar_chart(n_clicks):
            return self.fig

        def run_dash_server():
            # Start Dash app here without the reloader
            self.app.run_server(debug=False, use_reloader=False)

        self.dash_thread = threading.Thread(target=run_dash_server)
        self.dash_thread.start()

        if not args.disable_joint_setting_ui:
            import webbrowser
            webbrowser.open('http://127.0.0.1:8050/')


    def save_fig(self, save_path):
        self.fig.write_image(save_path)

    def create_tree_view_frame(self):
        frame = QtWidgets.QGroupBox("")
        layout = QtWidgets.QVBoxLayout()

        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.itemSelectionChanged.connect(self.on_tree_select)
        layout.addWidget(self.tree)
        
        frame.setLayout(layout)
        return frame

    def create_joint_list_frame(self):
        frame = QtWidgets.QGroupBox("")
        layout = QtWidgets.QVBoxLayout()

        self.joint_list = QtWidgets.QListWidget()
        self.joint_list.itemSelectionChanged.connect(self.joint_select)
        layout.addWidget(self.joint_list)

        frame.setLayout(layout)
        return frame
    
    def create_point_selection_frame(self):
        frame = QtWidgets.QGroupBox("")
        layout = QtWidgets.QVBoxLayout()

        # Create a PyVista plotter within the Qt window
        self.plotter = QtInteractor(frame)
        layout.addWidget(self.plotter.interactor)
        frame.setLayout(layout)

        # Load an STL file and add a sphere to the PyVista plotter
        self.load_model()

        return frame
    
    def create_point_selection_slider_frame(self):
        frame = QtWidgets.QGroupBox("Point Selection Slider")
        layout = QtWidgets.QVBoxLayout()

        # Create sliders for controlling X, Y, Z coordinates
        self.slider_x = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_y = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_z = QtWidgets.QSlider(QtCore.Qt.Horizontal)


        # Set slider ranges using self.mesh_bounds
        bound = self.mesh_bounds
        self.mesh_middle_point = [(bound[0] + bound[1]) / 2, (bound[2] + bound[3]) / 2, (bound[4] + bound[5]) / 2]
        self.slider_x.setRange(bound[0], bound[1])
        self.slider_y.setRange(bound[2], bound[3])
        self.slider_z.setRange(bound[4], bound[5])
        self.slider_x.setValue(self.mesh_middle_point[0])
        self.slider_y.setValue(self.mesh_middle_point[1])
        self.slider_z.setValue(self.mesh_middle_point[2])

        # Update the min, max of the joint_x_input based on the mesh bounds
        self.joint_x_input.setRange(bound[0], bound[1])
        self.joint_y_input.setRange(bound[2], bound[3])
        self.joint_z_input.setRange(bound[4], bound[5])

        # Add labels
        layout.addWidget(QtWidgets.QLabel("X Position"))
        layout.addWidget(self.slider_x)
        layout.addWidget(QtWidgets.QLabel("Y Position"))
        layout.addWidget(self.slider_y)
        layout.addWidget(QtWidgets.QLabel("Z Position"))
        layout.addWidget(self.slider_z)

        # Connect sliders to their respective functions
        self.slider_x.valueChanged.connect(self.slider_position_updated)
        self.slider_y.valueChanged.connect(self.slider_position_updated)
        self.slider_z.valueChanged.connect(self.slider_position_updated)

        frame.setLayout(layout)
        return frame


    def load_model(self):
        # Load STL file
        stl_file_path = self.args.stl_mesh_path
        stl_mesh = pv.read(stl_file_path)
        # Get the bounds of the mesh
        self.mesh_bounds = stl_mesh.bounds
        center = [(self.mesh_bounds[0] + self.mesh_bounds[1]) / 2, (self.mesh_bounds[2] + self.mesh_bounds[3]) / 2, (self.mesh_bounds[4] + self.mesh_bounds[5]) / 2]
        # Create a sphere (representing a point)
        self.shpere_position = center
        self.sphere = pv.Sphere(radius=1, center=self.shpere_position)
        # Add the STL mesh with transparency
        self.plotter.add_mesh(stl_mesh, opacity=0.5, color="lightblue")
        # Add a red sphere
        self.sphere_actor = self.plotter.add_mesh(self.sphere, color="red")
        self.plotter.add_axes()
        self.plotter.show()

    def slider_position_updated(self):
        """ Update the sphere position based on the slider values """
        x = self.slider_x.value()
        y = self.slider_y.value()
        z = self.slider_z.value()
        self.shpere_position = (x, y, z)
        self.update_sphere_position(self.shpere_position)

        self.joint_x_input.setValue(x)
        self.joint_y_input.setValue(y)
        self.joint_z_input.setValue(z)

    def update_slider_position(self):
        """ Update the sphere position based on the joint input values """
        x = self.joint_x_input.value()
        y = self.joint_y_input.value()
        z = self.joint_z_input.value()

        self.slider_x.setValue(x)
        self.slider_y.setValue(y)
        self.slider_z.setValue(z)
    
    def update_sphere_position(self, new_position):
        """ Update the sphere position and refresh the plot """
        self.sphere_position = new_position
        
        # Remove the old sphere
        self.plotter.remove_actor(self.sphere_actor)
        # Add a new sphere at the updated position
        self.sphere_actor = self.plotter.add_mesh(pv.Sphere(radius=1, center=self.sphere_position), color="red")
        # Update the plotter to reflect changes
        self.plotter.render()


    def create_link_edit_frame(self):
        frame = QtWidgets.QGroupBox("* Step 1: Link Edit")
        layout = QtWidgets.QVBoxLayout()

        self.link_name_input = QtWidgets.QLineEdit()
        self.combo_parent_name = QtWidgets.QComboBox()
        self.combo_parent_name.addItem("NONE")
        
        add_link_button = QtWidgets.QPushButton("Add Link")
        add_link_button.clicked.connect(self.add_link)
        remove_link_button = QtWidgets.QPushButton("Remove Link")
        remove_link_button.clicked.connect(self.remove_link)

        layout.addWidget(QtWidgets.QLabel("Input New Link Name"))
        layout.addWidget(self.link_name_input)
        layout.addWidget(QtWidgets.QLabel("Parent Name"))
        layout.addWidget(self.combo_parent_name)
        
        add_remove_hbox = QHBoxLayout()
        add_remove_hbox.addWidget(add_link_button)
        add_remove_hbox.addWidget(remove_link_button)
        layout.addLayout(add_remove_hbox)

        frame.setLayout(layout)
        return frame

    def create_joint_edit_frame(self):
        frame = QtWidgets.QGroupBox("* Step 2: Joint Edit")
        layout = QtWidgets.QVBoxLayout()

        self.combo_joint_name = QtWidgets.QComboBox()
        self.combo_joint_name.setEditable(True)
        self.combo_joint_name.addItem("No_name")
        self.combo_joint_name.currentTextChanged.connect(self.joint_combo_select)

        self.joint_x_input = QtWidgets.QDoubleSpinBox()
        self.joint_y_input = QtWidgets.QDoubleSpinBox()
        self.joint_z_input = QtWidgets.QDoubleSpinBox()
        self.joint_x_input.setRange(-1e6, 1e6)
        self.joint_y_input.setRange(-1e6, 1e6)
        self.joint_z_input.setRange(-1e6, 1e6)

        # Set step size for more granular control
        self.joint_x_input.setSingleStep(0.1)
        self.joint_y_input.setSingleStep(0.1)
        self.joint_z_input.setSingleStep(0.1)

        # Set the number of decimal places to show
        self.joint_x_input.setDecimals(2)
        self.joint_y_input.setDecimals(2)
        self.joint_z_input.setDecimals(2)

        self.joint_x_input.valueChanged.connect(self.update_slider_position)
        self.joint_y_input.valueChanged.connect(self.update_slider_position)
        self.joint_z_input.valueChanged.connect(self.update_slider_position)

        layout.addWidget(QtWidgets.QLabel("Joint Name"))
        layout.addWidget(self.combo_joint_name)

        hbox_x = QHBoxLayout()
        hbox_x.addWidget(QLabel("Position X"))
        hbox_x.addWidget(self.joint_x_input)
        layout.addLayout(hbox_x)

        hbox_y = QHBoxLayout()
        hbox_y.addWidget(QLabel("Position Y"))
        hbox_y.addWidget(self.joint_y_input)
        layout.addLayout(hbox_y)

        hbox_z = QHBoxLayout()
        hbox_z.addWidget(QLabel("Position Z"))
        hbox_z.addWidget(self.joint_z_input)
        layout.addLayout(hbox_z)

        add_remove_hbox = QHBoxLayout()
        add_joint_button = QtWidgets.QPushButton("Add Joint")
        add_joint_button.clicked.connect(self.add_joint)
        remove_joint_button = QtWidgets.QPushButton("Remove Joint")
        remove_joint_button.clicked.connect(self.remove_joint)
        add_remove_hbox.addWidget(add_joint_button)
        add_remove_hbox.addWidget(remove_joint_button)
        layout.addLayout(add_remove_hbox)
        
        frame.setLayout(layout)
        return frame

    def create_axis_edit_frame(self):
        frame = QtWidgets.QGroupBox("* Step 3: Axis Edit")
        layout = QtWidgets.QVBoxLayout()

        self.axis_display = QtWidgets.QLabel("Current Axis:")
        self.axis_input = QtWidgets.QLineEdit()
        add_axis_button = QtWidgets.QPushButton("Add Axis")
        add_axis_button.clicked.connect(self.add_axis)
        remove_axis_button = QtWidgets.QPushButton("Remove Axis")
        remove_axis_button.clicked.connect(self.remove_axis)

        layout.addWidget(self.axis_display)
        layout.addWidget(self.axis_input)
        explain_text = QtWidgets.QLabel("Format:[(3d position) + one or two (3d directions)]")
        explain_text2 = QtWidgets.QLabel("E.g. [(-0.1, 0, 0), (0, 1, 0)]")
        font = QtGui.QFont()
        font.setPointSize(8)  # Set the font size to 16 points
        explain_text.setFont(font)
        explain_text2.setFont(font)

        layout.addWidget(explain_text)
        layout.addWidget(explain_text2)

        hbox = QHBoxLayout()
        hbox.addWidget(add_axis_button)
        hbox.addWidget(remove_axis_button)
        layout.addLayout(hbox)

        frame.setLayout(layout)
        return frame

    def create_save_frame(self):
        frame = QtWidgets.QGroupBox("Save")
        layout = QtWidgets.QVBoxLayout()

        save_button = QtWidgets.QPushButton("Save")
        save_button.clicked.connect(self.save)
        layout.addWidget(save_button)

        start_design_button = QtWidgets.QPushButton("Run Design Process")
        start_design_button.clicked.connect(self.start_design)
        layout.addWidget(start_design_button)

        frame.setLayout(layout)
        return frame


    def update_joint_list(self):
        self.joint_list.clear()
        if self.current_link:
            for joint_name, joint_position in self.current_link.joints.items():
                self.joint_list.addItem(f"{joint_name}: {joint_position}")

    def shutdown(self):
        self.server.shutdown()
        self.server_thread.join()
        self.dash_thread.join()
        self.close()

    # Overriding the closeEvent method
    def closeEvent(self, event):
        """Customize the action when the window's 'X' button is clicked."""
        if not self.start_design_flag:
            self.shutdown()
            exit(0)
        else:
            self.shutdown()

    def start_design(self):
        reply = QtWidgets.QMessageBox.question(self, 'Start', 
                                               "Do you want to quit the UI and start the design process?", 
                                               QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, 
                                               QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            self.start_design_flag = True
            self.shutdown()

    def save(self):
        # Save the nodes as a pickle file
        pkl.dump(self.nodes, open(f'./auto_design/model/given_models/{self.args.model_name}_joints.pkl', 'wb'))

        # Save a copy in result folder
        pkl.dump(self.nodes, open(f'{self.args.result_folder}/{self.args.model_name}_joints.pkl', 'wb'))

        # Confirm save
        reply = QtWidgets.QMessageBox.question(self, 'Save', 
                                               "Save successful. Start the design process?", 
                                               QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, 
                                               QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            self.start_design_flag = True
            self.shutdown()

    def remove_link(self):
        selected_items = self.tree.selectedItems()
        if not selected_items:
            QtWidgets.QMessageBox.warning(self, "No link selected", "Please select a link to remove.")
            return
        
        selected_item = selected_items[0].text(0)
        
        # Remove the selected link from the nodes
        if selected_item in self.nodes:
            # Recursively remove the selected link's children
            self.recursive_children_remove(selected_item)

            # Remove the selected link.
            self.remove_from_children(selected_item)
            self.nodes[selected_item].val.joints = {}
            del self.nodes[selected_item]

            # Refresh the joint list
            self.joint_list.clear()
            for joint_name, joint_position in self.current_link.joints.items():
                self.joint_list.addItem(f"{joint_name}: {joint_position}")

            self.tree.clear()
            self.load_tree(self.nodes)

            self.update_plot()

            self.update_parent_name_combobox()
            self.update_joint_combobox()
            self.axis_input.setText("None")

    def recursive_children_remove(self, selected_item):
        children_copy = list(self.nodes[selected_item].children)
        for child in children_copy:
            self.recursive_children_remove(child.val.name)

            # Remove the selected link name from the links who has the selected link as a child
            self.remove_from_children(child.val.name)
            self.nodes[child.val.name].val.joints = {}
            del self.nodes[child.val.name]
            print(f"Removed {child.val.name} from children of {selected_item}")


    def remove_from_children(self, selected_item):
        # Remove the selected link name from the links who has the selected link as a child
        if selected_item in self.nodes:
            for node in self.nodes.values():
                for child in node.children:
                    if child.val.name == selected_item:
                        node.children.remove(child)
                        print(f"Removed {selected_item} from {node.val.name}'s children.")

    def add_link(self):
        link_name = self.link_name_input.text()
        parent_name = self.combo_parent_name.currentText()
        
        if link_name and link_name not in self.nodes:
            link = Link(link_name)
            node = TreeNode(link)

            if parent_name in self.nodes:
                self.nodes[link_name] = node 
                self.nodes[parent_name].add_child(node)
            else:
                # ROOT NODE
                if link_name != "BODY":
                    QtWidgets.QMessageBox.warning(self, "ROOT NODE CAN ONLY BE BODY", "ADDING BODY as the root node.")
                    link_name = "BODY"

                if link_name not in self.nodes:
                    link = Link(link_name)
                    node = TreeNode(link)
                    self.nodes[link_name] = node
                
            self.link_name_input.clear()
            self.load_tree(self.nodes)

            # Update combobox
            self.combo_parent_name.addItem(link_name)
        else:
            QtWidgets.QMessageBox.warning(self, "No link name or link already exists", "Please enter a new link name or the link already exists.")

    def joint_select(self):
        selected_items = self.joint_list.selectedItems()
        if selected_items:
            joint_name, joint_position = selected_items[0].text().split(":")
            joint_position = joint_position.strip()

            joint_pos = ast.literal_eval(joint_position)
            self.combo_joint_name.setCurrentText(joint_name)

            self.joint_x_input.setValue(joint_pos[0])
            self.joint_y_input.setValue(joint_pos[1])
            self.joint_z_input.setValue(joint_pos[2])

    def joint_combo_select(self):
        joint_name = self.combo_joint_name.currentText()
        for link in self.nodes.values():
            if joint_name in link.val.joints:
                joint_pos = link.val.joints[joint_name]
                self.joint_x_input.setValue(joint_pos[0])
                self.joint_y_input.setValue(joint_pos[1])
                self.joint_z_input.setValue(joint_pos[2])
                break

    def on_tree_select(self):
        selected_items = self.tree.selectedItems()
        if selected_items:
            selected_item = selected_items[0].text(0)
            self.current_link = self.nodes[selected_item].val if selected_item in self.nodes else None
            self.update_joint_list()

            # Update combobox
            self.combo_parent_name.setCurrentText(selected_item)

            #Update axis display
            if self.current_link and self.current_link.axis:
                text_to_display = ""
                for elements in self.current_link.axis:
                    text_to_display += str(elements)
                    text_to_display += ","
                # Change "[]" to "()"
                text_to_display = text_to_display.replace("[", "(")
                text_to_display = text_to_display.replace("]", ")")
                # Remove the last comma
                text_to_display = text_to_display[:-1] + "]"
                text_to_display = "[" + text_to_display
                self.axis_input.setText(text_to_display)
            else:
                self.axis_input.setText("None")


    def add_joint(self):
        if self.current_link:
            joint_name = self.combo_joint_name.currentText()
            
            if joint_name == "No_name" or not joint_name:
                QtWidgets.QMessageBox.warning(self, "No joint name", "Please enter or select a joint name.")
                return
            
            x, y, z = self.joint_x_input.value(), self.joint_y_input.value(), self.joint_z_input.value()
                
            self.current_link.add_joint(joint_name, (x, y, z))
            
            # Check if the joint is already in another link and if the position is different
            for link in self.nodes.values():
                if joint_name in link.val.joints:
                    if np.any(link.val.joints[joint_name] != (x, y, z)):
                        print(f"Joint already exists in another link with a different position.")
                        print(f"Current joint position: {x, y, z}")
                        print(f"Existing joint position: {link.val.joints[joint_name]}")
                        
                        message_to_print = (f"Joint {joint_name} already exists in {link.val.name} with "
                                            f"an existing position: {link.val.joints[joint_name]}. "
                                            "Do you want to overwrite the position?")
                        
                        reply = QtWidgets.QMessageBox.question(self, "Overwrite", message_to_print, 
                                                               QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
                        if reply == QtWidgets.QMessageBox.Yes:
                            self.update_joint(link.val.name, joint_name, (x, y, z))
                        else:
                            continue
            
            self.update_plot()

            # Refresh the joint list
            self.joint_list.clear()
            for joint_name, joint_position in self.current_link.joints.items():
                self.joint_list.addItem(f"{joint_name}: {joint_position}")
        else:
            QtWidgets.QMessageBox.warning(self, "No link selected", "Please select a link to add the joint to.")

    def update_joint(self, link_name, joint_name, joint_position):
        if joint_name in self.nodes[link_name].val.joints:
            self.nodes[link_name].val.add_joint(joint_name, joint_position)
            # Reconstruct the joint lines
            self.nodes[link_name].val.construct_joint_lines()
        else:
            print("Joint not found in the link.")
        print(self.nodes[link_name].val.joints)

    def remove_joint(self):
        if self.current_link:
            selected_items = self.joint_list.selectedItems()
            if selected_items:
                joint_name = selected_items[0].text().split(":")[0]
                
                # Remove the line made by the joint
                for line in self.current_link.joint_lines:
                    if np.all(line.start == self.current_link.joints[joint_name]) or np.all(line.end == self.current_link.joints[joint_name]):
                        self.current_link.joint_lines.remove(line)
                
                # Remove the joint
                del self.current_link.joints[joint_name]
                self.update_plot()

                # Refresh the joint list
                self.joint_list.clear()
                for joint_name, joint_position in self.current_link.joints.items():
                    self.joint_list.addItem(f"{joint_name}: {joint_position}")
            else:
                QtWidgets.QMessageBox.warning(self, "No joint selected", "Please select a joint to remove.")
        else:
            QtWidgets.QMessageBox.warning(self, "No link selected", "Please select a link to remove the joint from.")

    def add_axis(self):
        if self.current_link:
            axis_str = self.axis_input.text()
            #axis = tuple(map(float, axis_str.split(",")))
            try:
                # Try to safely convert the string to a list of tuples
                axis_str = axis_str.strip() # Remove leading/trailing whitespace
                real_list = ast.literal_eval(axis_str)

                # Ensure the result is actually a list of tuples
                if isinstance(real_list, list) and all(isinstance(item, tuple) for item in real_list):
                    # Turn real_list into a one-dimensional list
                    real_list = [item for sublist in real_list for item in sublist]
                    print(real_list)

                    self.current_link.add_axis(real_list)
                    self.update_plot()
                else:
                    QtWidgets.QMessageBox.warning(self, "Invalid axis format", "Please enter a valid axis format.")
            except (SyntaxError, ValueError) as e:
                print(f"Error: {e}")
                QtWidgets.QMessageBox.warning(self, "Invalid axis format", "Please enter a valid axis format.")
        else:
            QtWidgets.QMessageBox.warning(self, "No link selected", "Please select a link to add the axis to.")

    def remove_axis(self):
        if self.current_link:
            self.current_link.axis = None
            self.update_plot()
            self.axis_input.setText("None")
        else:
            QtWidgets.QMessageBox.warning(self, "No link selected", "Please select a link to remove the axis from.")

    def update_plot(self):
        self.fig.data = []  # Clear existing data
        x, y, z = [], [], []
        cone_size = 10
        axis_x, axis_y, axis_z, direct_x, direct_y, direct_z = [], [], [], [], [], []

        for link in self.nodes.values():
            if link.val is None or link.val.axis is None:
                print(f"Warning: link {link.val} has no value or axis")
                continue

            if len(link.val.axis) == 2:
                axis_x.append(link.val.axis[0][0])
                axis_y.append(link.val.axis[0][1])
                axis_z.append(link.val.axis[0][2])
                direct_x.append(link.val.axis[1][0] * cone_size)
                direct_y.append(link.val.axis[1][1] * cone_size)
                direct_z.append(link.val.axis[1][2] * cone_size)
            elif len(link.val.axis) == 3:
                axis_x.append(link.val.axis[0][0])
                axis_y.append(link.val.axis[0][1])
                axis_z.append(link.val.axis[0][2])
                direct_x.append(link.val.axis[1][0] * cone_size)
                direct_y.append(link.val.axis[1][1] * cone_size)
                direct_z.append(link.val.axis[1][2] * cone_size)

                axis_x.append(link.val.axis[0][0])
                axis_y.append(link.val.axis[0][1])
                axis_z.append(link.val.axis[0][2])
                direct_x.append(link.val.axis[2][0] * cone_size)
                direct_y.append(link.val.axis[2][1] * cone_size)
                direct_z.append(link.val.axis[2][2] * cone_size)

            for pos in link.val.joints.values():
                x.append(pos[0])
                y.append(pos[1])
                z.append(pos[2])

        # Add joint markers and axes to the plot
        self.fig.add_trace(self.mesh.mesh_plotly)
        self.fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers'))
        cone = go.Cone(x=axis_x, y=axis_y, z=axis_z, u=direct_x, v=direct_y, w=direct_z)
        self.fig.add_trace(cone)
        # Optionally refresh the plot if required


    def get_tree(self):
        """Function to return the root node of the tree."""
        return self.nodes.get("BODY", None)

    def load_tree(self, nodes):
        """Function to load a tree structure from the provided nodes."""
        self.nodes = nodes
        print(self.nodes)

        self.tree.clear()

        # Add "BODY" as the root node in the tree widget
        body_item = QtWidgets.QTreeWidgetItem(["BODY"])
        self.tree.addTopLevelItem(body_item)
        inserted_items_set = set()
        inserted_items_set.add("BODY")

        # Recursively add children to the root node
        root_node = self.nodes.get("BODY", None)
        self.add_children_to_tree(body_item, root_node)

        # Expand all nodes by default
        self.tree.expandAll()

        # Ensure joint lines are constructed
        for node_name, node in nodes.items():
            node.val.construct_joint_lines()
            
        # Update the parent name combobox with the node names
        self.update_parent_name_combobox()

        # Update the joint combobox with the joint names
        self.update_joint_combobox()


    def add_children_to_tree(self, parent_item, node):
        """Recursively add child nodes to the tree."""
        for child in node.children:
            # Create a child tree item
            child_item = QtWidgets.QTreeWidgetItem([child.val.name])

            # Add child to the parent
            parent_item.addChild(child_item)

            # Recursively add children's children if they exist
            if child.children:
                self.add_children_to_tree(child_item, child)


    def update_parent_name_combobox(self):
        """Helper function to update the parent name combobox with node names."""
        self.combo_parent_name.clear()
        self.combo_parent_name.addItem("NONE")  # Reset with "NONE"
        
        for node in self.nodes.values():
            self.combo_parent_name.addItem(node.val.name)

    def update_joint_combobox(self):
        """Helper function to update the joint name combobox with available joints."""
        self.combo_joint_name.clear()
        self.combo_joint_name.addItem("No_name")  # Reset with "No_name"

        for node in self.nodes.values():
            for joint_name in node.val.joints:
                if joint_name not in [self.combo_joint_name.itemText(i) for i in range(self.combo_joint_name.count())]:
                    self.combo_joint_name.addItem(joint_name)



class Mesh_Loader:
    def __init__(self, args):
        self.args = args
        self.scaled_mesh = None
        self.scaled_joint_dict = {}
        self.joint_dict = {}
        self.link_tree = None
    
    def load_mesh(self, mesh_path : str):
        """
        Load the mesh data from the file system.
        """
        self.mesh = Mesh(mesh_path)
        return self.mesh
    
    def load_joint_positions(self, joint_path : str):
        """
        Load the joint data from the file system.
        """
        pass
    
    def set_scale(self):
        """
        Do the preprocess to scale the mesh and joint data.
        """
        pass

    
    def scale(self, expected_x, save_path=None):
        """
        Scale the mesh, joint data, and link tree according to the expected x-axis length.
        """

        # Get the scale factor
        vertices = np.asarray(self.mesh.mesh_o3d.vertices)
        self.scale_factor = expected_x / 2 / (np.max(vertices[:,0]))

        # Scale the mesh
        self.scaled_mesh = self.mesh
        self.scaled_mesh.scale(self.scale_factor)

        # save the scaled mesh if save_path is provided
        if save_path is not None:
            self.scaled_mesh.mesh_o3d.compute_vertex_normals()  # Compute normals
            o3d.io.write_triangle_mesh(save_path, self.scaled_mesh.mesh_o3d)

        # Scale the joint data
        for joint_name in self.joint_dict:
            self.scaled_joint_dict[joint_name] = np.array(self.joint_dict[joint_name]) * self.scale_factor

        # self.scaled_joint_dict = {joint_name: joint_position * self.scale_factor for joint_name, joint_position in self.joint_dict.items()}

        # Update the link tree

        if self.link_tree is not None:

            for joint_name in self.link_tree.val.joints:
                self.link_tree.val.joints[joint_name] = np.array(self.link_tree.val.joints[joint_name]) * self.scale_factor

            self.link_tree.val.axis = list(self.link_tree.val.axis)
            self.link_tree.val.axis[0] = np.array(self.link_tree.val.axis[0]) * self.scale_factor
            
            for link in self.link_tree.get_all_children()[0]:
                for joint_name in link.val.joints:
                    link.val.joints[joint_name] = np.array(link.val.joints[joint_name]) * self.scale_factor

                link.val.axis = list(link.val.axis)
                link.val.axis[0] = np.array(link.val.axis[0]) * self.scale_factor

    def update_link_tree(self):
        pass
        

    def render(self, save_only=False, save_path=None):
        """
        Render the scaled mesh and joint data.
        """
        axes_lines = create_axes_lines(np.array([0,0,0]), 
                                       np.array([1,0,0]), 
                                       np.array([0,1,0]), 
                                       np.array([0,0,1]))
        transformed_joints_vis, transformed_lines_vis = create_joint_visualization(self.scaled_joint_dict)
        fig = go.Figure(data=[transformed_joints_vis, self.mesh.mesh_plotly, *transformed_lines_vis, *axes_lines])
        fig.update_layout(
            autosize=False,
            margin = {'l':0,'r':0,'t':0,'b':0},
            scene=dict(
                xaxis=dict(showgrid=False, showticklabels=False, backgroundcolor="rgba(0,0,0,0)", 
                        zeroline=False, showbackground=False, title=''),  # Remove X-axis title
                yaxis=dict(showgrid=False, showticklabels=False, backgroundcolor="rgba(0,0,0,0)",
                        zeroline=False, showbackground=False, title=''),  # Remove Y-axis title
                zaxis=dict(showgrid=False, showticklabels=False, backgroundcolor="rgba(0,0,0,0)",
                        zeroline=False, showbackground=False, title=''),  # Remove Z-axis title
            ),
            scene_aspectmode='data',
            # plot_bgcolor='rgba(0,0,0,0)',  # Set plot background to be transparent
            # paper_bgcolor='rgba(0,0,0,0)',  # Set paper background to be transparent
            showlegend=False,  # Hide the legend
            annotations=[],  # Remove annotations
            scene_camera=dict(up=dict(x=0, y=0, z=1), center=dict(x=0, y=-0.25, z=0), eye=dict(x=1.2, y=-1.0, z=0.4)),  # Optional: Adjust camera for better view
            width=740,
            height=600
        )
        #fig.show()
        if not save_only:
            fig.show()
        if save_path is not None:
            fig.write_image(save_path)
        print("scale factor:", self.scale_factor)

    def run(self, render=True):
        """
        The main function to run the mesh loader.
        """
        self.mesh_dir = os.path.normpath('./model/sample_models/' + self.args.model_name + '_res_e300_smoothed.stl')
        self.joint_dir = os.path.normpath('./model/sample_models/' + self.args.model_name + '_joints.npy')
        self.load_mesh(self.mesh_dir)
        self.load_joint_positions(self.joint_dir)
        self.set_scale()
        self.scale()
        if render:
            self.render()



class Custom_Mesh_Loader(Mesh_Loader):
    def __init__(self, args):
        super().__init__(args)

    def load_joint_positions(self, joint_path: str, figure_save_path=None):
        
        # Create an instance of QApplication
        app = QApplication(sys.argv)
        if os.path.exists(joint_path):
            print("Loading joint data from file...")
            with open(joint_path, 'rb') as f:
                #linkLoader = LinkTreeGUI(tk.Tk(), self.mesh, self.args)
                linkLoader = LinkTreeGUI(self.mesh, self.args)
                linkLoader.nodes = pkl.load(f)
                linkLoader.load_tree(linkLoader.nodes)
                linkLoader.update_plot()

                linkLoader.show()
                print(linkLoader.nodes)
                
                # Shutdown the GUI immediately if the joint data is already provided and the joint setting UI is disabled
                if self.args.disable_joint_setting_ui:
                    time.sleep(1)
                    linkLoader.shutdown()
                
        else:
            print("No joint data found. Please construct the link tree.")
            # linkLoader = LinkTreeGUI(tk.Tk(), self.mesh, self.args)
            linkLoader = LinkTreeGUI(self.mesh, self.args)
            linkLoader.show()
        
        # exit the application
        app.exec_()

        #linkLoader.root.mainloop()

        self.link_tree = linkLoader.get_tree()

        # Get all joint positions
        self.joint_dict = {}

        for joint_name in self.link_tree.val.joints:
            self.joint_dict[joint_name] = self.link_tree.val.joints[joint_name]

        self.link_tree.val.construct_joint_lines()
        for link in self.link_tree.get_all_children()[0]:
            link.val.construct_joint_lines()
            for joint_name, joint_position in link.val.joints.items():
                self.joint_dict[joint_name] = joint_position
        
        # Save the joint data if the figure_save_path is provided
        if figure_save_path is not None:
            linkLoader.save_fig(figure_save_path)
            

    def render(self, save_only=False, save_path=None):
        fig = go.Figure()

        x, y, z = [], [], []
        for joint in self.scaled_joint_dict.items():
            x.append(joint[1][0])
            y.append(joint[1][1])
            z.append(joint[1][2])
        
        # Add joint markers
        fig.add_trace(self.scaled_mesh.mesh_plotly)
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers'))

        if not save_only:
            fig.show(renderer="browser")
        if save_path is not None:
            fig.write_image(save_path)
        #fig.show(renderer="browser")  # Refresh the plot in the browser window

class Quadruped_Mesh_Loader(Mesh_Loader):
    def __init__(self, args):
        super().__init__(args)
    
    def load_joint_positions(self, joint_path : str):
        """
        Load the joint data from the file system.
        """
        joint_data = np.load(joint_path)
        joint_dict = {
            "waist": joint_data[4],
            "hip": joint_data[1],
            "left_hip" : joint_data[17],
            "right_hip":joint_data[21],
            "left_knee": joint_data[18],
            "right_knee":joint_data[22],
            "left_ankle":joint_data[19],
            "right_ankle":joint_data[23],
            "scapula":joint_data[6],
            "left_shoulder":joint_data[7],
            "right_shoulder":joint_data[11],
            "left_elbow":joint_data[8],
            "right_elbow":joint_data[12],
            "left_wrist":joint_data[9],
            "right_wrist":joint_data[13],
            'left_hand':joint_data[10],
            'right_hand':joint_data[14],
            'left_foot':joint_data[20],
            'right_foot':joint_data[24],
            'head':joint_data[16],
            'tail':joint_data[30],
        }

        # Adjust the joint positions:
        ## 1. Push the hip joints down
        joint_dict["left_hip"] = (joint_dict["left_hip"] + joint_dict["left_knee"]) / 2
        joint_dict["right_hip"] = (joint_dict["right_hip"] + joint_dict["right_knee"]) / 2
        
        ## 2. Push the hip, shoulder, elbow, and knee joints towards the center
        center_hip = (joint_dict["left_hip"] + joint_dict["right_hip"]) / 2
        center_shoulder = (joint_dict["left_shoulder"] + joint_dict["right_shoulder"]) / 2
        center_elbow = (joint_dict["left_elbow"] + joint_dict["right_elbow"]) / 2
        center_knee = (joint_dict["left_knee"] + joint_dict["right_knee"]) / 2
        joint_dict["left_hip"] = joint_dict["left_hip"] + (center_hip - joint_dict["left_hip"]) * 0.3
        joint_dict["right_hip"] = joint_dict["right_hip"] + (center_hip - joint_dict["right_hip"]) * 0.3
        joint_dict["left_shoulder"] = joint_dict["left_shoulder"] + (center_shoulder - joint_dict["left_shoulder"]) * 0.3
        joint_dict["right_shoulder"] = joint_dict["right_shoulder"] + (center_shoulder - joint_dict["right_shoulder"]) * 0.3
        joint_dict["left_elbow"] = joint_dict["left_elbow"] + (center_elbow - joint_dict["left_elbow"]) * 0.15
        joint_dict["right_elbow"] = joint_dict["right_elbow"] + (center_elbow - joint_dict["right_elbow"]) * 0.15
        joint_dict["left_knee"] = joint_dict["left_knee"] + (center_knee - joint_dict["left_knee"]) * 0.15
        joint_dict["right_knee"] = joint_dict["right_knee"] + (center_knee - joint_dict["right_knee"]) * 0.15
        self.joint_dict = joint_dict

    def update_link_tree(self):
        # Update Link Tree
        joint_dict = self.scaled_joint_dict

        # Define the link configuration
        link_config = {
            "BODY": ["waist", "hip", "scapula", "head", "tail"],
            "FL_UP": ["left_shoulder", "left_elbow"],
            "FL_LOW": ["left_elbow", "left_wrist", "left_hand"],
            "FR_UP": ["right_shoulder", "right_elbow"],
            "FR_LOW": ["right_elbow", "right_wrist", "right_hand"],
            "RL_UP": ["left_hip", "left_knee"],
            "RL_LOW": ["left_knee", "left_ankle", "left_foot"],
            "RR_UP": ["right_hip", "right_knee"],
            "RR_LOW": ["right_knee", "right_ankle", "right_foot"]
        }

        links = {}
        nodes = {}

        # Create links and tree nodes
        for link_name, joints in link_config.items():
            link = Link(link_name)
            link.add_joints({joint: joint_dict[joint] for joint in joints})
            links[link_name] = link
            nodes[link_name] = TreeNode(link)

        # Define parent-child relationships
        hierarchy = {
            "BODY": ["FL_UP", "FR_UP", "RL_UP", "RR_UP"],
            "FL_UP": ["FL_LOW"],
            "FR_UP": ["FR_LOW"],
            "RL_UP": ["RL_LOW"],
            "RR_UP": ["RR_LOW"]
        }

        # Build the tree structure
        for parent, children in hierarchy.items():
            for child in children:
                nodes[parent].add_child(nodes[child])

        self.link_tree = nodes["BODY"]

    def set_scale(self):
        """
        Do the preprocess to scale the mesh and joint data.
        """
        # Set Transformation
        ## 1. The new origin is the projection of the waist joint on the ground plane
        self.new_origin = np.array([self.joint_dict["waist"][0], 
                                    self.joint_dict["waist"][1], 
                                    min(self.mesh.mesh_plotly.z)])
        ## 2. The new axis(y is the heading direction from waist to hip, z is the vertical direction)
        body_direct = (self.joint_dict["hip"] - self.joint_dict["waist"]) / np.linalg.norm(self.joint_dict["hip"] - self.joint_dict["waist"])
        self.new_z_axis = np.array([0, 0, 1])
        self.new_x_axis = np.cross(body_direct, self.new_z_axis)
        self.new_y_axis = np.cross(self.new_z_axis, self.new_x_axis)

        ## 3. The transformation matrix
        self.transformation = np.block([[np.column_stack([self.new_x_axis, self.new_y_axis, self.new_z_axis]), self.new_origin.reshape((3, 1))],
                                        [np.array([0, 0, 0, 1])]])
        self.transformation = np.linalg.inv(self.transformation)
        
        return self.transformation

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Mesh Loader')
    # parser.add_argument('--model_name', type=str, default='jkhk', help='The model name')
    # parser.add_argument('--expected_x', type=float, default=12.5, help='The expected width of the model')
    # args = parser.parse_args()
    # mesh_loader = Quadruped_Mesh_Loader(args)
    # mesh_dir = os.path.normpath('./model/sample_models/' + args.model_name + '_res_e300_smoothed.stl')
    # joint_dir = os.path.normpath('./model/sample_models/' + args.model_name + '_joints.npy')
    # mesh_loader.load_mesh(mesh_dir)
    # mesh_loader.load_joint_positions(joint_dir)
    # mesh_loader.set_scale()
    # mesh_loader.scale()
    # mesh_loader.render()

    parser = argparse.ArgumentParser(description='Mesh Loader')
    parser.add_argument('--model_name', type=str, default='lynel', help='The model name')
    parser.add_argument('--expected_x', type=float, default=40, help='The expected width of the model')
    args = parser.parse_args()
    mesh_loader = Custom_Mesh_Loader(args)
    mesh_dir = os.path.normpath('./auto_design/model/given_models/' + args.model_name + '.stl')
    joint_dir = os.path.normpath('./auto_design/model/given_models/' + args.model_name + '_joints.pkl')
    mesh_loader.load_mesh(mesh_dir)
    mesh_loader.load_joint_positions(joint_dir)
    # print(mesh_loader.link_tree.get_all_children())