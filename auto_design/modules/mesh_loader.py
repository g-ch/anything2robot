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
from tkinter import ttk
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output
import requests
from wsgiref.simple_server import make_server
from flask import Flask

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
        return np.linalg.norm(closest - point)
    
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

    
    def render(self):
        """
        Render the mesh data.
        """
        fig = go.Figure(data=[self.mesh_plotly])
        fig.show()
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
class LinkTreeGUI:
    def __init__(self, root, mesh, args):
        self.args = args
        self.root = root
        self.root.title("Link Tree Constructor")
        
        self.nodes = {}
        self.current_link = None
        
        # Layout configuration
        self.frame = ttk.Frame(self.root, padding="3 3 12 12")
        self.frame.grid(column=0, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Tree view
        self.tree = ttk.Treeview(self.frame)
        self.tree.grid(column=0, row=0, columnspan=3, sticky='nsew')
        self.tree.bind("<<TreeviewSelect>>", self.on_tree_select)

        # Joints in current link
        self.joint_list = tk.Listbox(self.frame)
        self.joint_list.grid(column=3, row=0, columnspan=4, sticky='nsew')
        self.joint_list.bind('<<ListboxSelect>>',self.joint_select)

        # Axis of current link
        self.cur_rotation_axis = tk.StringVar()
        ttk.Label(self.frame, text="Rotation Axis:").grid(column=5, row=1)
        # Add text box to display rotation axis
        ttk.Label(self.frame, textvariable=self.cur_rotation_axis).grid(column=6, row=1)
        # ttk.Text(self.frame, textvariable=self.cur_rotation_axis, width=30).grid(column=6, row=1, columnspan=2)
        
        # Add link controls
        ttk.Label(self.frame, text="Link Name:").grid(column=0, row=1)
        self.link_name = tk.StringVar()
        ttk.Entry(self.frame, textvariable=self.link_name).grid(column=1, row=1)
        
        ttk.Label(self.frame, text="Parent Name:").grid(column=0, row=2)
        self.parent_name = tk.StringVar()
        ttk.Entry(self.frame, textvariable=self.parent_name).grid(column=1, row=2)
        self.rotation_axis = tk.StringVar()
        ttk.Label(self.frame, text="Axis:").grid(column=0, row=3)
        ttk.Entry(self.frame, textvariable=self.rotation_axis).grid(column=1, row=3)
        
        
        ttk.Button(self.frame, text="Add Link", command=self.add_link).grid(column=0, row=4, columnspan=2)
        ttk.Button(self.frame, text="Remove Link", command=self.remove_link).grid(column=0, row=5, columnspan=2)
        ttk.Button(self.frame, text="Add Axis", command=self.add_axis).grid(column=0, row=6, columnspan=2)

        
        # Joint controls
        ttk.Label(self.frame, text="Joint Name:").grid(column=2, row=1)
        self.joint_name = tk.StringVar()
        ttk.Entry(self.frame, textvariable=self.joint_name).grid(column=3, row=1)
        
        ttk.Label(self.frame, text="X:").grid(column=2, row=2)
        self.joint_x = tk.DoubleVar()
        ttk.Entry(self.frame, textvariable=self.joint_x).grid(column=3, row=2)
        
        ttk.Label(self.frame, text="Y:").grid(column=2, row=3)
        self.joint_y = tk.DoubleVar()
        ttk.Entry(self.frame, textvariable=self.joint_y).grid(column=3, row=3)
        
        ttk.Label(self.frame, text="Z:").grid(column=2, row=4)
        self.joint_z = tk.DoubleVar()
        ttk.Entry(self.frame, textvariable=self.joint_z).grid(column=3, row=4)
        
        ttk.Button(self.frame, text="Add Joint", command=self.add_joint).grid(column=2, row=5, columnspan=2)
        ttk.Button(self.frame, text="Remove Joint", command=self.remove_joint).grid(column=2, row=6, columnspan=2)

        # quit button
        ttk.Button(self.frame, text="Quit", command=self.quit).grid(column=0, row=7, columnspan=4)

        # save button
        ttk.Button(self.frame, text="Save", command=self.save).grid(column=0, row=8, columnspan=4)

        # Plotly visualization
        self.fig = make_subplots(specs=[[{"type": "scene"}]])
        self.fig.add_trace(mesh.mesh_plotly)
        self.mesh = mesh

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
        

        def run_dash():
            self.app.run_server(debug=True)
        # Create a thread to run the Dash app
        self.dash_thread = Thread(target=run_dash)
        self.dash_thread.start()
        import webbrowser
        webbrowser.open('http://127.0.0.1:8050/')

    def quit(self):
        self.server.shutdown()
        self.server_thread.join()
        self.dash_thread.join()
        self.root.quit()

    def save(self):
        pkl.dump(self.nodes, open('./auto_design/model/given_models/' + self.args.model_name + '_joints.pkl', 'wb'))

    def remove_link(self):
        selected_item = self.tree.selection()[0]
        if selected_item in self.nodes:
            del self.nodes[selected_item]
            self.tree.delete(selected_item)
            for node in self.nodes.values():
                for child in node.children:
                    if child.val.name == selected_item:
                        node.children.remove(child)
                        break

    def add_link(self):
        link_name = self.link_name.get()
        parent_name = self.parent_name.get()
        if link_name and link_name not in self.nodes:
            link = Link(link_name)
            node = TreeNode(link)
            self.nodes[link_name] = node
            if parent_name and parent_name in self.nodes:
                self.nodes[parent_name].add_child(node)
                parent_id = self.nodes[parent_name].val.name
                self.tree.insert(parent_id, 'end', link_name, text=link_name)
            else:
                self.tree.insert('', 'end', link_name, text=link_name)
            self.link_name.set("")
            self.parent_name.set("")
    
    def joint_select(self, event):
        selected_joint = self.joint_list.curselection()
        joint_name = self.joint_list.get(selected_joint[0]).split(":")[0]
        joint_pos = self.joint_list.get(selected_joint[0]).split(":")[1]
        self.joint_name.set(joint_name)
        tuple_val = ast.literal_eval(joint_pos[1:])
        joint_pos = list(tuple_val)

        self.joint_x.set(joint_pos[0])
        self.joint_y.set(joint_pos[1])
        self.joint_z.set(joint_pos[2])


    def on_tree_select(self, event):
        selected_item = self.tree.selection()[0]
        self.current_link = self.nodes[selected_item].val if selected_item in self.nodes else None

        # Update joint list
        self.joint_list.delete(0, tk.END)
        if self.current_link:
            for joint_name, joint_position in self.current_link.joints.items():
                self.joint_list.insert(tk.END, f"{joint_name}: {joint_position}")
        
        # Update rotation axis
        if self.current_link.axis:
            
            renderings = ""
            renderings_modify = ""
            for i, axis in enumerate(self.current_link.axis):
                if i == 0:
                    renderings += f"Origin: {axis}\n"
                    renderings_modify += ','.join(map(str, axis)) + ','
                else:
                    renderings += f"Axis {i}: {axis}\n"
                    renderings_modify += ','.join(map(str, axis)) + ','
            self.cur_rotation_axis.set(renderings)
            self.rotation_axis.set(renderings_modify[:-1])
        else:
            self.cur_rotation_axis.set("")

    def add_joint(self):
        if self.current_link:
            joint_name = self.joint_name.get()
            x, y, z = self.joint_x.get(), self.joint_y.get(), self.joint_z.get()
            self.current_link.add_joint(joint_name, (x, y, z))
            self.update_plot()
    
    def add_axis(self):
        if self.current_link:
            axis_str = self.rotation_axis.get()
            axis = tuple(map(float, axis_str.split(",")))
            self.current_link.add_axis(axis)
            self.update_plot()
    
    def remove_joint(self):
        if self.current_link:
            selected_joint = self.joint_list.curselection()
            if selected_joint:
                joint_name = self.joint_list.get(selected_joint[0]).split(":")[0]
                
                # remove the line made by the joint
                for line in self.current_link.joint_lines:
                    if line.start == self.current_link.joints[joint_name] or line.end == self.current_link.joints[joint_name]:
                        self.current_link.joint_lines.remove(line)
                
                # remove the joint
                del self.current_link.joints[joint_name]
                self.update_plot()

    def update_plot(self):
        self.fig.data = []  # Clear existing data
        x, y, z = [], [], []
        cone_size = 10
        axis_x, axis_y, axis_z, direct_x, direct_y, direct_z = [], [], [], [], [], []
        for link in self.nodes.values():
            if len(link.val.axis) == 2:
                axis_x.append(link.val.axis[0][0])
                axis_y.append(link.val.axis[0][1])
                axis_z.append(link.val.axis[0][2])
                direct_x.append(link.val.axis[1][0]*cone_size)
                direct_y.append(link.val.axis[1][1]*cone_size)
                direct_z.append(link.val.axis[1][2]*cone_size)
            elif len(link.val.axis) == 3:
                axis_x.append(link.val.axis[0][0])
                axis_y.append(link.val.axis[0][1])
                axis_z.append(link.val.axis[0][2])
                direct_x.append(link.val.axis[1][0]*cone_size)
                direct_y.append(link.val.axis[1][1]*cone_size)
                direct_z.append(link.val.axis[1][2]*cone_size)

                axis_x.append(link.val.axis[0][0])
                axis_y.append(link.val.axis[0][1])
                axis_z.append(link.val.axis[0][2])
                direct_x.append(link.val.axis[2][0]*cone_size)
                direct_y.append(link.val.axis[2][1]*cone_size)
                direct_z.append(link.val.axis[2][2]*cone_size)

            for pos in link.val.joints.values():
                x.append(pos[0])
                y.append(pos[1])
                z.append(pos[2])
        
        # Add joint markersplotly 
        self.fig.add_trace(self.mesh.mesh_plotly)
        self.fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers'))
        cone = go.Cone(x=axis_x, y=axis_y, z=axis_z, u=direct_x, v=direct_y, w=direct_z)
        self.fig.add_trace(cone)
        # self.fig.show(renderer="browser")  # Refresh the plot in the browser window
    
    def get_tree(self):
        return self.nodes["BODY"]
    
    def load_tree(self, nodes):
        self.nodes = nodes
        self.tree.insert('', 'end', "BODY", text="BODY")
        for node_name, node in nodes.items():
            node.val.construct_joint_lines()
            parent_name = node_name
            for child in node.children:
                self.tree.insert(parent_name, 'end', child.val.name, text=child.val.name)
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

    
    def scale(self):
        """
        Scale the mesh, joint data, and link tree according to the expected x-axis length.
        """

        # Get the scale factor
        vertices = np.asarray(self.mesh.mesh_o3d.vertices)
        self.scale_factor = self.args.expected_x / 2 / (np.max(vertices[:,0]))

        # Scale the mesh
        self.scaled_mesh = self.mesh
        self.scaled_mesh.scale(self.scale_factor)

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
        


    def render(self):
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
        fig.show()
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

    def load_joint_positions(self, joint_path: str):
        
        if os.path.exists(joint_path):
            print("Loading joint data from file...")
            with open(joint_path, 'rb') as f:
                linkLoader = LinkTreeGUI(tk.Tk(), self.mesh, self.args)
                linkLoader.nodes = pkl.load(f)
                linkLoader.load_tree(linkLoader.nodes)
                linkLoader.update_plot()
                print(linkLoader.nodes)
                
        else:
            print("No joint data found. Please construct the link tree.")
            linkLoader = LinkTreeGUI(tk.Tk(), self.mesh, self.args)
        
        linkLoader.root.mainloop()
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

    def render(self):
        fig = go.Figure()

        x, y, z = [], [], []
        for joint in self.scaled_joint_dict.items():
            x.append(joint[1][0])
            y.append(joint[1][1])
            z.append(joint[1][2])
        
        # Add joint markers
        fig.add_trace(self.scaled_mesh.mesh_plotly)
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers'))
        fig.show(renderer="browser")  # Refresh the plot in the browser window

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