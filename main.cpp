#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_MultiFab.H>
#include <AMReX_EB2_IF_Cylinder.H>
#include <AMReX_EBFArrayBox.H>
#include <AMReX_EBFabFactory.H>
#include <AMReX_EB2.H>
#include <AMReX_EB2_IF.H>
#include <AMReX_EB_utils.H>
#include <solveManager.H>

int main (int argc, char* argv[])
{
    amrex::Initialize(argc,argv);

    {
        solveManager solver;
        solver.read_problem_inputs();
        solver.init_fields();

        amrex::ParmParse pp;

        Real time=0.0;
        Real dt=0.001;
        int max_step=1;
        int step=0;
        Real tfinal;

        pp.get("timestep",dt);
        pp.get("max_step",max_step);
        pp.get("final_time",tfinal);
        
        solver.write_outputs(step,time);

        while((time<tfinal) and (step<max_step))
        {
                solver.set_bc_vals(time);
                solver.update_staggered_layers();
                solver.vel_solve(time,dt);
                solver.update_staggered_layers();
                solver.update_divergence_of_vel();
                solver.pres_poisson();
                solver.project_vel();
                solver.update_staggered_layers();
                solver.update_divergence_of_vel();
                time += dt;
                step += 1;
        }
        solver.write_outputs(step,time);
    }

    amrex::Finalize();
}
