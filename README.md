# nadi (river)

## Incompressible staggered grid solver using amrex

### How to?

* After cloning, do $git submodule init && git submodule update
to get the latest amrex.

* You will need gcc and openmpi/mpich library to build.

* Go to, the tests/drivenCavity and do make -j. It should give you an executable - nadi3d.gnu.MPI.ex.
* If you want to try running on GPUs, do make -j USE_CUDA=TRUE. You will also need nvcc along with gcc and mpi library.

* running - mpirun -n \<nprocs\> nadi3d.gnu.MPI.ex inputs. It does 3000 steps, takes about a minute with 32 processors.
  
* verify solution - use python script verify_vel.py python verify_vel.py <last_plot_folder>, you will get an image file called vel_drivencavity_x.png.
