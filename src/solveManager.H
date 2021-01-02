#ifndef SOLVEMANAGER_H
#define SOLVEMANAGER_H

#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_MultiFab.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_BCRec.H>
#include <AMReX_MultiFabUtil.H>

using namespace amrex;

class solveManager
{
    private:

        MultiFab *m_pres;
        MultiFab *m_vx;
        MultiFab *m_vy;
        MultiFab *m_vz;
        MultiFab *m_divu;

        MultiFab *m_cc_flow_vars;

        Geometry m_geom;
        BoxArray m_ba;
        DistributionMapping m_dmap;
        int m_ng, m_max_grid_size;
        Vector<Real> m_plo;
        Vector<Real> m_phi;
        Vector<int> m_ncells;
        Real m_dx[3];
        Real m_visc;

        Vector<int> m_bc_lo;
        Vector<int> m_bc_hi;

        Vector<BCRec> m_bcrecs;
        BCRec m_vy_bc;
        BCRec m_vz_bc;
        BCRec m_pres_bc;

    public:

        solveManager(){};
        void read_problem_inputs();
        void write_outputs(int step,Real time);
        void init_fields();
        void update_cc_vars();
        void update_staggered_layers();
        void set_bc_vals(Real time);
        void pres_poisson();
        void update_divergence_of_vel();
        void project_vel();
        void vel_solve(Real time,Real dt); 
        ~solveManager();
};
#endif
