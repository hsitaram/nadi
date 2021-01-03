#include<solve_manager.H>
#include<initial_conditions.H>
#include<global_defines.H>
#include<boundary_conditions.H>
#include <AMReX_MLMG.H>
#include <AMReX_MLPoisson.H>

//=======================================================================
void solveManager::read_problem_inputs()
{
    m_ng = 1;
    m_max_grid_size=64;

    ParmParse pp;
    pp.getarr("prob_lo",m_plo);
    pp.getarr("prob_hi",m_phi);
    pp.getarr("ncells",m_ncells);
    pp.query("max_grid_size",m_max_grid_size);
    pp.get("viscosity",m_visc);

    Vector<int> is_periodic(AMREX_SPACEDIM,0);
    pp.getarr("is_it_periodic",is_periodic);

    RealBox real_box({AMREX_D_DECL(m_plo[0], m_plo[1], m_plo[2])},
            {AMREX_D_DECL(m_phi[0], m_phi[1], m_phi[2])});

    IntVect domain_lo(AMREX_D_DECL(0,0,0));
    IntVect domain_hi(AMREX_D_DECL(m_ncells[0]-1,m_ncells[1]-1,m_ncells[2]-1));

    m_dx[0]=(m_phi[0]-m_plo[0])/m_ncells[0];
    m_dx[1]=(m_phi[1]-m_plo[1])/m_ncells[1];
    m_dx[2]=(m_phi[2]-m_plo[2])/m_ncells[2];

    Box domain(domain_lo, domain_hi);
    m_ba.define(domain);
    m_ba.maxSize(m_max_grid_size);

    m_geom.define(domain,&real_box,CoordSys::cartesian,is_periodic.data());
    m_dmap.define(m_ba);

    BoxArray facex_ba = amrex::convert(m_ba, {1,0,0});
    BoxArray facey_ba = amrex::convert(m_ba, {0,1,0});
    BoxArray facez_ba = amrex::convert(m_ba, {0,0,1});

    m_pres         = new MultiFab(m_ba,m_dmap,1,m_ng);
    m_divu         = new MultiFab(m_ba,m_dmap,1,0);
    m_vx           = new MultiFab(facex_ba,m_dmap,1,m_ng);
    m_vy           = new MultiFab(facey_ba,m_dmap,1,m_ng);
    m_vz           = new MultiFab(facez_ba,m_dmap,1,m_ng);

    //one extra for div
    m_cc_flow_vars = new MultiFab(m_ba,m_dmap,FLOW_VARS+1,m_ng);

    //boundaries
    m_bc_lo.resize(AMREX_SPACEDIM);
    m_bc_hi.resize(AMREX_SPACEDIM);
    
    pp.queryarr("lo_bc", m_bc_lo, 0, AMREX_SPACEDIM);
    pp.queryarr("hi_bc", m_bc_hi, 0, AMREX_SPACEDIM);
}
//=======================================================================
void solveManager::write_outputs(int step,Real time)
{
    std::string pltfile_prefix,pltfname;
    pltfile_prefix="plt";
    pltfname=amrex::Concatenate(pltfile_prefix, step, 5);

    const Array<const MultiFab *,AMREX_SPACEDIM> allvel={m_vx,m_vy,m_vz};

    MultiFab::Copy(*m_cc_flow_vars, *m_pres, 0, PRES_INDX, 1, 0);
    average_face_to_cellcenter(*m_cc_flow_vars,VELX_INDX,allvel);
    MultiFab::Copy(*m_cc_flow_vars, *m_divu, 0, VELZ_INDX+1, 1, 0);

    Vector<std::string> varnames={"pressure","velx","vely","velz","divu"};
    WriteSingleLevelPlotfile(pltfname, *m_cc_flow_vars, varnames, m_geom, time, step);

    //local cleanup
    varnames.clear();
}
//=======================================================================
void solveManager::init_fields()
{
    const auto dx = m_geom.CellSizeArray();
    auto prob_lo  = m_geom.ProbLoArray();

    for(MFIter mfi(*m_pres); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.validbox();
        auto vx_arr=(*m_vx)[mfi].array();
        auto vy_arr=(*m_vy)[mfi].array();
        auto vz_arr=(*m_vz)[mfi].array();
        auto pr_arr=(*m_pres)[mfi].array();

        Box x_bx=convert(bx, {1,0,0});
        Box y_bx=convert(bx, {0,1,0});
        Box z_bx=convert(bx, {0,0,1});

        amrex::ParallelFor(x_bx, [=] 
          AMREX_GPU_DEVICE(int i,int j,int k)
        {
            set_vx_ic(i,j,k,vx_arr,prob_lo,dx);   
        });
        amrex::ParallelFor(y_bx, [=] 
          AMREX_GPU_DEVICE(int i,int j,int k)
        {
            set_vy_ic(i,j,k,vy_arr,prob_lo,dx);   
        });
        amrex::ParallelFor(z_bx, [=] 
          AMREX_GPU_DEVICE(int i,int j,int k)
        {
            set_vz_ic(i,j,k,vz_arr,prob_lo,dx);   
        });
        amrex::ParallelFor(bx, [=] 
            AMREX_GPU_DEVICE(int i,int j,int k)
        {
            set_pres_ic(i,j,k,pr_arr,prob_lo,dx);   
        });
    }
}
//=======================================================================
solveManager::~solveManager()
{
    delete(m_pres);
    delete(m_vx);
    delete(m_vy);
    delete(m_vz);
    delete(m_cc_flow_vars);
    delete(m_divu);

    m_plo.clear();
    m_phi.clear();
    m_ncells.clear();
    m_bc_lo.clear();
    m_bc_hi.clear();
}
//=======================================================================
