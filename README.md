# nadi
incompressible staggered grid solver using amrex

How to?

after cloning, do perform git submodule init and git submodule update to get the latest amrex
You will need gcc and openmpi/mpich library to build
Go to, the tests/drivenCavity and do make -j. It should give you an executable - nadi3d.gnu.MPI.ex
run - mpirun -n <nprocs> nadi3d.gnu.MPI.ex inputs
verify solution - use python script verify_vel.py python verify_vel.py <last_plot_folder>, you will get an image file called vel_drivencavity_x.png
