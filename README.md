# anything2robot


## Installation
__Tested Environment: Ubuntu 20.04__

- Create a conda environment
```
conda create --name anything2robot python=3.8
conda activate anything2robot
python3 -m pip install --user open3d
```

Please make sure that the version of open3d is 0.18.0.

- Install openscad and solidpython. Run the following command
```
sudo apt-get install openscad
pip install solidpython
```

- Install CGAL. By default this will install CGAL 5.0.2-3. when using Ubuntu 20.04.
```
sudo apt-get install libcgal-dev
```

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
