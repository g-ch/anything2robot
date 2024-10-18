import numpy as np
import copy
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from matplotlib import animation

class Contact_Calculator():

    def __init__(self) -> None:
        self.rotation_resolution = 1
        self.contact_idx = 0

    def calculate_com(self, points):
        """ Calculate the center of mass of the points. """
        return np.mean(points, axis=0)

    def rotate_points(self, points, axis, angle):
        """ Rotate points around a given axis by a certain angle in degrees. """
        rot = R.from_rotvec(axis * np.deg2rad(angle))
        return rot.apply(points)

    def find_contact_point(self, points, axis, start_points):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim([0, 2])
        ax.set_ylim([0, 2])
        ax.set_zlim([0, 2])

        def update(angle):
            """ Update function for the animation. """
            self.rotated_points = self.rotate_points(points, axis, angle)
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='r', marker='o')
            ax.scatter(self.rotated_points[:, 0], self.rotated_points[:, 1], self.rotated_points[:, 2], c='b', marker='o', s=100)
            contact_idxs = np.where(self.rotated_points[:, 2] < 0, 1, 0)
            for contact_idx, flag in enumerate(contact_idxs):
                if flag and not np.any([np.all(np.isclose(self.rotated_points[contact_idx], start_point, atol=2e-2)) for start_point in start_points]):
                    print(self.rotated_points[:, 2])
                    print(f"Contact at angle {angle} degrees.")
                    self.contact_idx = contact_idx
                    ani.event_source.stop()
                    break

        ani = animation.FuncAnimation(fig, update, frames=np.linspace(0, 360, int(360 // self.rotation_resolution)), repeat=False, interval=0.1)
        plt.show()

        return self.contact_idx, np.round(self.rotated_points, 2)

    def run(self, points):
        transformed_points = copy.deepcopy(points)
        idx_a = np.argmin(points[:, 2])
        transformed_points[:, 2] = transformed_points[:, 2] - np.min(points[:, 2])
        point_a = transformed_points[idx_a]
        
        # Rotate around cross product of (A - CoM) and z-axis
        cross_prod = np.cross(point_a - self.calculate_com(transformed_points), [0, 0, 1])
        if np.linalg.norm(cross_prod) == 0:
            return [point_a]

        # Find point B
        idx_b, transformed_points = self.find_contact_point(transformed_points, cross_prod, [point_a])
        if idx_b is None:
            print("No second contact point found, maybe set a smaller threshold")
            return None
        point_b = transformed_points[idx_b]

        # Find point C
        projection = np.cross(self.calculate_com(transformed_points) - point_a, point_b - point_a)[2]
        if projection == 0:
            return [point_a, point_b]
        line_ab = point_b - point_a if projection > 0 else point_a - point_b
        idx_c, transformed_points = self.find_contact_point(transformed_points, line_ab, [point_a, point_b])
        if idx_c is None:
            print("No third contact point found.")
            return None
        point_c = transformed_points[idx_c]

        contact_points = np.array([point_a, point_b, point_c])
        M_bary = np.array([[contact_points[0][0], contact_points[1][0], contact_points[2][0]],
                           [contact_points[0][1], contact_points[1][1], contact_points[2][1]],
                           [1,                    1,                    1]])
        CoM = self.calculate_com(transformed_points)
        param_bary = np.linalg.inv(M_bary) @ np.array([CoM[0], CoM[1], 1])

        # Rotate around line A-B to find C until CoM's projection is within contact points
        while(np.any(param_bary < 0)):
            print(param_bary)
            
            rm_idxs = np.where(param_bary < 0, 1, 0)
            
            if np.sum(rm_idxs) == 2:
            # Start with finding the second point
                point_a = contact_points[rm_idxs==0][0]
                cross_prod = np.cross(point_a - self.calculate_com(transformed_points), [0, 0, 1])

                # Find point B
                idx_b, transformed_points = self.find_contact_point(transformed_points, cross_prod, [point_a])
                if idx_b is None:
                    print("No second contact point found, maybe set a smaller threshold")
                    return None
                point_b = transformed_points[idx_b]
            
            else:
                point_a = contact_points[rm_idxs==0][0]
                point_b = contact_points[rm_idxs==0][1]
            
            # self.render(transformed_points, [CoM])
            # print(contact_points, CoM)
            # print(np.abs(contact_points - CoM))

            # Recalculate point C
            projection = np.cross(self.calculate_com(transformed_points) - point_a, point_b - point_a)[2]
            if projection == 0:
                return [point_a, point_b]
            line_ab = point_b - point_a if projection > 0 else point_a - point_b
            idx_c, transformed_points = self.find_contact_point(transformed_points, line_ab, [point_a, point_b])
            if idx_c is None:
                print("No third contact point found.")
                return None
            point_c = transformed_points[idx_c]

            contact_points = np.array([point_a, point_b, point_c])
            M_bary = np.array([[contact_points[0][0], contact_points[1][0], contact_points[2][0]],
                               [contact_points[0][1], contact_points[1][1], contact_points[2][1]],
                               [1,                    1,                    1]])
            CoM = self.calculate_com(transformed_points)
            param_bary = np.linalg.inv(M_bary) @ np.array([CoM[0], CoM[1], 1])
            
        self.render(transformed_points, [CoM])
        return [point_a, point_b, point_c]
    
    def render(self, points, contact_points=[], save_path=None, save_only=False):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='r', marker='o')

        # Highlight contact points
        for point in contact_points:
            ax.scatter(point[0], point[1], point[2], c='b', marker='o', s=100)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_zlim3d(0, 2)                    # viewrange for z-axis should be [-4,4] 
        ax.set_ylim3d(0, 2)                    # viewrange for y-axis should be [-2,2] 
        ax.set_xlim3d(0, 2)                    # viewrange for x-axis should be [-2,2] 
        
        #plt.show()
        if not save_only:
            plt.show()
        if save_path is not None:
            plt.savefig(save_path, dpi=300)
            plt.close()


# Example usage with an example set of points
points = np.array([
    [0, 0, 0],
    [1, 1, 1],
    [0.2, 0.5, 0.1],
    [0, 1, 0]
])

contact_cal = Contact_Calculator()

result = contact_cal.run(points)
if result:
    print("Supporting Points A, B, C:", result)
else:
    print("Simulation did not find a stable configuration.")

# contact_cal.render(points, result)
