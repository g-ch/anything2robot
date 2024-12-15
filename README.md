# anything2robot

## Installation (Using Mamba)
__Tested Environment: Ubuntu 20.04__

* If you don't have Mamba installed, refer to Fresh installation in [Page](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html).

- Create a Mamba environment and activate
```
mamba create --name anything2robot python=3.9
mamba activate anything2robot
```

- Install Open3d and scikit-learn. Please make sure that the version of open3d is 0.18.0.
```
python3 -m pip install --user open3d==0.18.0 scikit-learn==1.3.2 scikit-image
```

- Install UI related libs
```
pip install -U kaleido==0.2.1 vtk pyvista trimesh
mamba install conda-forge::qt anaconda::pyqt conda-forge::pyvistaqt
```

- Install openscad, CGAL, solidpython. Run the following command
```
sudo apt-get install openscad libcgal-dev
pip install solidpython
```
By default this will install CGAL 5.0.2-3. when using Ubuntu 20.04. A different CGAL version may result in error.


- Install meshio by
```
pip install meshio[all]
```

- Install pinochhio
```
conda install conda-forge::pinocchio==2.7.0
```

- Install Ansys 2023 R2 (Note: Student edition can't be installed in Ubuntu. Make sure you have the license). To install Ansys on ubuntu, first, install the dependencies shown below. Then follow the official instructions of Ansys to install it in Ubuntu. __NOTE__: Use the default ```/root/ansys_inc``` folder to install. Install only MAPDL is sufficient. Finally, install PyAnsy. 
```
sudo apt install ubuntu-desktop alien freeglut3 libxcb-xinerama0 lsb xterm libmotif-common  libmotif-dev
```
```
Install MAPDL (Structural Mechanics) on your ubuntu system. Use the default installation path otherwise the installation may fail.
```
```
python -m pip install pyansys
```


- Clone the code to your PC and compile the C++ part.

```
git clone git@github.com:Anything-robot/anything2robot.git

cd anything2robot/metamaterial_filling
mkdir build
cd build
cmake ..
make
```

- Make a data folder. Put your stl file/folder in the data folder.
```
cd metamaterial_filling
mkdir data
```


## Installation (Using Conda)
__Tested Environment: Ubuntu 20.04__

- Create a conda environment
```
conda create --name anything2robot python=3.8
conda activate anything2robot
python3 -m pip install --user open3d==0.18.0 scikit-learn
```
* Please make sure that the version of open3d is 0.18.0.


- Install UI related libs
```
conda install conda-forge::qt conda-forge::pyvistaqt
```

- Install openscad, CGAL, solidpython. Run the following command
```
sudo apt-get install openscad libcgal-dev
pip install solidpython
```
By default this will install CGAL 5.0.2-3. when using Ubuntu 20.04. A different CGAL version may result in error.


- Install meshio by
```
pip install meshio[all]
```

- Install pinochhio
```
python -m pip install pin
```

- Install Ansys 2023 R2 (Note: Student edition can't be installed in Ubuntu. Make sure you have the license). To install Ansys on ubuntu, first, install the dependencies shown below. Then follow the official instructions of Ansys to install it in Ubuntu. __NOTE__: Use the default ```/root/ansys_inc``` folder to install. Install only MAPDL is sufficient. Finally, install PyAnsy.
```
sudo apt install ubuntu-desktop alien freeglut3 libxcb-xinerama0 lsb xterm libmotif-common  libmotif-dev
```
```
Install MAPDL on your ubuntu system
```
```
python -m pip install pyansys
```

- Then clone the code to your PC and compile the C++ part.

```
git clone git@github.com:Anything-robot/anything2robot.git

cd anything2robot/metamaterial_filling
mkdir build
cd build
cmake ..
make
```

- Make a data folder. Put your stl file/folder in the data folder.
```
cd metamaterial_filling
mkdir data
```


