#include<solveManager.H>
#include<initialize_fields.H>
#include<globalDefines.H>
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

    m_bcrecs.resize(FLOW_VARS);

    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
        // lo-side BCs
        if (m_bc_lo[idim] == WALL_ID)
        {
           m_bcrecs[PRES_INDX].setLo(idim,BCType::reflect_even);
           m_bcrecs[VELX_INDX].setLo(idim,BCType::reflect_odd);
           m_bcrecs[VELY_INDX].setLo(idim,BCType::reflect_odd);
           m_bcrecs[VELZ_INDX].setLo(idim,BCType::reflect_odd);
        }
        else if(m_bc_lo[idim] == INFLOW_ID)
        {
           m_bcrecs[PRES_INDX].setLo(idim,BCType::reflect_even);
           m_bcrecs[VELX_INDX].setLo(idim,BCType::ext_dir);
           m_bcrecs[VELY_INDX].setLo(idim,BCType::ext_dir);
           m_bcrecs[VELZ_INDX].setLo(idim,BCType::ext_dir);
        }
        else if(m_bc_lo[idim] == OUTFLOW_ID)
        {
           m_bcrecs[PRES_INDX].setLo(idim,BCType::ext_dir);
           m_bcrecs[VELX_INDX].setLo(idim,BCType::foextrap);
           m_bcrecs[VELY_INDX].setLo(idim,BCType::foextrap);
           m_bcrecs[VELZ_INDX].setLo(idim,BCType::foextrap);
        }
        else if(m_bc_lo[idim] == PERIODIC_ID)
        {
           m_bcrecs[PRES_INDX].setLo(idim,BCType::int_dir);
           m_bcrecs[VELX_INDX].setLo(idim,BCType::int_dir);
           m_bcrecs[VELY_INDX].setLo(idim,BCType::int_dir);
           m_bcrecs[VELZ_INDX].setLo(idim,BCType::int_dir);
        } 
        else 
        {
            amrex::Abort("Invalid bc_lo");
        }
        
        // hi-side BCs
        if (m_bc_hi[idim] == WALL_ID)
        {
           m_bcrecs[PRES_INDX].setLo(idim,BCType::reflect_even);
           m_bcrecs[VELX_INDX].setLo(idim,BCType::reflect_odd);
           m_bcrecs[VELY_INDX].setLo(idim,BCType::reflect_odd);
           m_bcrecs[VELZ_INDX].setLo(idim,BCType::reflect_odd);
        }
        else if(m_bc_hi[idim] == INFLOW_ID)
        {
           m_bcrecs[PRES_INDX].setLo(idim,BCType::reflect_even);
           m_bcrecs[VELX_INDX].setLo(idim,BCType::ext_dir);
           m_bcrecs[VELY_INDX].setLo(idim,BCType::ext_dir);
           m_bcrecs[VELZ_INDX].setLo(idim,BCType::ext_dir);
        }
        else if(m_bc_hi[idim] == OUTFLOW_ID)
        {
           m_bcrecs[PRES_INDX].setLo(idim,BCType::ext_dir);
           m_bcrecs[VELX_INDX].setLo(idim,BCType::foextrap);
           m_bcrecs[VELY_INDX].setLo(idim,BCType::foextrap);
           m_bcrecs[VELZ_INDX].setLo(idim,BCType::foextrap);
        }
        else if(m_bc_hi[idim] == PERIODIC_ID)
        {
           m_bcrecs[PRES_INDX].setLo(idim,BCType::int_dir);
           m_bcrecs[VELX_INDX].setLo(idim,BCType::int_dir);
           m_bcrecs[VELY_INDX].setLo(idim,BCType::int_dir);
           m_bcrecs[VELZ_INDX].setLo(idim,BCType::int_dir);
        } 
        else 
        {
            amrex::Abort("Invalid bc_hi");
        }
    }

}
//=======================================================================
void solveManager::set_bc_vals(Real time)
{
    const auto dx    = m_geom.CellSizeArray();
    auto prob_lo     = m_geom.ProbLoArray();
    const int* domlo = m_geom.Domain().loVect();
    const int* domhi = m_geom.Domain().hiVect();

    int *bclo = m_bc_lo.data();
    int *bchi = m_bc_hi.data();

    int wall     = WALL_ID;
    int inflow   = INFLOW_ID;
    int outflow  = OUTFLOW_ID;
    int periodic = PERIODIC_ID;
    Real t       = time;
    
    (*m_vx).FillBoundary(m_geom.periodicity());
    (*m_vy).FillBoundary(m_geom.periodicity());
    (*m_vz).FillBoundary(m_geom.periodicity());

    for(MFIter mfi(*m_pres); mfi.isValid(); ++mfi)
    {
        const Box& bx  = mfi.validbox();
        const Box &gbx = amrex::grow(bx, m_ng);
        
        auto vx_arr=(*m_vx)[mfi].array();
        auto vy_arr=(*m_vy)[mfi].array();
        auto vz_arr=(*m_vz)[mfi].array();

        auto pr_arr=(*m_pres)[mfi].array();

        Box x_bx=convert(gbx, {1,0,0});
        Box y_bx=convert(gbx, {0,1,0});
        Box z_bx=convert(gbx, {0,0,1});

        amrex::ParallelFor(gbx, [=] AMREX_GPU_DEVICE(int i,int j,int k)
        {
            if(i==(domlo[0]-1) and bclo[0]!=periodic)
            {
                apply_pres_bc(i,j,k,pr_arr,t,prob_lo,dx,XLOFACE);
            }
            if(i==(domhi[0]+1) and bchi[0]!=periodic)
            {
                apply_pres_bc(i,j,k,pr_arr,t,prob_lo,dx,XHIFACE);
            }
            if(j==(domlo[1]-1) and bclo[1]!=periodic)
            {
                apply_pres_bc(i,j,k,pr_arr,t,prob_lo,dx,YLOFACE);
            }
            if(j==(domhi[1]+1) and bchi[1]!=periodic)
            {
                apply_pres_bc(i,j,k,pr_arr,t,prob_lo,dx,YHIFACE);
            }
            if(k==(domlo[2]-1) and bclo[2]!=periodic)
            {
                apply_pres_bc(i,j,k,pr_arr,t,prob_lo,dx,ZLOFACE);
            }
            if(k==(domhi[2]+1) and bchi[2]!=periodic)
            {
                apply_pres_bc(i,j,k,pr_arr,t,prob_lo,dx,ZHIFACE);
            }
        });

        amrex::ParallelFor(x_bx, [=] AMREX_GPU_DEVICE(int i,int j,int k)
        {       
            if(i==(domlo[0]-1) and bclo[0]!=periodic)
            {
                apply_vx_bc(i,j,k,vx_arr,pr_arr,t,prob_lo,dx,XLOFACE);
            }
            if(i==(domhi[0]+2) and bchi[0]!=periodic)
            {
                apply_vx_bc(i,j,k,vx_arr,pr_arr,t,prob_lo,dx,XHIFACE);
            }
            if(j==(domlo[1]-1) and bclo[1]!=periodic)
            {
                apply_vx_bc(i,j,k,vx_arr,pr_arr,t,prob_lo,dx,YLOFACE);
            }
            if(j==(domhi[1]+1) and bchi[1]!=periodic)
            {
                apply_vx_bc(i,j,k,vx_arr,pr_arr,t,prob_lo,dx,YHIFACE);
            }
            if(k==(domlo[2]-1) and bclo[2]!=periodic)
            {
                apply_vx_bc(i,j,k,vx_arr,pr_arr,t,prob_lo,dx,ZLOFACE);
            }
            if(k==(domhi[2]+1) and bchi[2]!=periodic)
            {
                apply_vx_bc(i,j,k,vx_arr,pr_arr,t,prob_lo,dx,ZHIFACE);
            }
        });

        amrex::ParallelFor(y_bx, [=] 
                AMREX_GPU_DEVICE(int i,int j,int k)
        {
            if(i==(domlo[0]-1) and bclo[0]!=periodic)
            {
                apply_vy_bc(i,j,k,vy_arr,pr_arr,t,prob_lo,dx,XLOFACE);
            }
            if(i==(domhi[0]+1) and bchi[0]!=periodic)
            {
                apply_vy_bc(i,j,k,vy_arr,pr_arr,t,prob_lo,dx,XHIFACE);
            }
            if(j==(domlo[1]-1) and bclo[1]!=periodic)
            {
                apply_vy_bc(i,j,k,vy_arr,pr_arr,t,prob_lo,dx,YLOFACE);
            }
            if(j==(domhi[1]+2) and bchi[1]!=periodic)
            {
                apply_vy_bc(i,j,k,vy_arr,pr_arr,t,prob_lo,dx,YHIFACE);
            }
            if(k==(domlo[2]-1) and bclo[2]!=periodic)
            {
                apply_vy_bc(i,j,k,vy_arr,pr_arr,t,prob_lo,dx,ZLOFACE);
            }
            if(k==(domhi[2]+1) and bchi[2]!=periodic)
            {
                apply_vy_bc(i,j,k,vy_arr,pr_arr,t,prob_lo,dx,ZHIFACE);
            }
        });

        amrex::ParallelFor(z_bx, [=] 
                AMREX_GPU_DEVICE(int i,int j,int k)
        {
            if(i==(domlo[0]-1) and bclo[0]!=periodic)
            {
                apply_vz_bc(i,j,k,vz_arr,pr_arr,t,prob_lo,dx,XLOFACE);
            }
            if(i==(domhi[0]+1) and bchi[0]!=periodic)
            {
                apply_vz_bc(i,j,k,vz_arr,pr_arr,t,prob_lo,dx,XHIFACE);
            }
            if(j==(domlo[1]-1) and bclo[1]!=periodic)
            {
                apply_vz_bc(i,j,k,vz_arr,pr_arr,t,prob_lo,dx,YLOFACE);
            }
            if(j==(domhi[1]+1) and bchi[1]!=periodic)
            {
                apply_vz_bc(i,j,k,vz_arr,pr_arr,t,prob_lo,dx,YHIFACE);
            }
            if(k==(domlo[2]-1) and bclo[2]!=periodic)
            {
                apply_vz_bc(i,j,k,vz_arr,pr_arr,t,prob_lo,dx,ZLOFACE);
            }
            if(k==(domhi[2]+2) and bchi[2]!=periodic)
            {
                apply_vz_bc(i,j,k,vz_arr,pr_arr,t,prob_lo,dx,ZHIFACE);
            }
        });

   }
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
void solveManager::update_divergence_of_vel()
{
    const auto dx = m_geom.CellSizeArray();

    for(MFIter mfi(*m_divu); mfi.isValid(); ++mfi)
    {
        const Box& bx  = mfi.validbox();
        
        auto vx_arr=(*m_vx)[mfi].array();
        auto vy_arr=(*m_vy)[mfi].array();
        auto vz_arr=(*m_vz)[mfi].array();

        auto rhs_arr=(*m_divu)[mfi].array();

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i,int j,int k)
        { 
           rhs_arr(i,j,k)  = (vx_arr(i+1,j,k)-vx_arr(i,j,k))/dx[0];
           rhs_arr(i,j,k) += (vy_arr(i,j+1,k)-vy_arr(i,j,k))/dx[1];
           rhs_arr(i,j,k) += (vz_arr(i,j,k+1)-vz_arr(i,j,k))/dx[2];
        });
   }

   Print()<<"div vel:"<<m_divu->norm0()<<"\n";
   PrintToFile("velnorms")<<m_vx->norm0()
       <<"\t"<<m_vy->norm0()<<"\t"<<m_vz->norm0()<<"\n";
}
//=======================================================================
void solveManager::project_vel()
{
    const auto dx = m_geom.CellSizeArray();
    (*m_pres).FillBoundary(m_geom.periodicity());

    for(MFIter mfi(*m_pres); mfi.isValid(); ++mfi)
    {
        const Box& bx  = mfi.validbox();
        
        auto vx_arr=(*m_vx)[mfi].array();
        auto vy_arr=(*m_vy)[mfi].array();
        auto vz_arr=(*m_vz)[mfi].array();

        auto pr_arr=(*m_pres)[mfi].array();

        Box x_bx=convert(bx, {1,0,0});
        Box y_bx=convert(bx, {0,1,0});
        Box z_bx=convert(bx, {0,0,1});

        amrex::ParallelFor(x_bx, [=] AMREX_GPU_DEVICE(int i,int j,int k)
        {
            vx_arr(i,j,k) -= (pr_arr(i,j,k)-pr_arr(i-1,j,k))/dx[0];
        });

        amrex::ParallelFor(y_bx, [=] AMREX_GPU_DEVICE(int i,int j,int k)
        {
            vy_arr(i,j,k) -= (pr_arr(i,j,k)-pr_arr(i,j-1,k))/dx[1];
        });

        amrex::ParallelFor(z_bx, [=] AMREX_GPU_DEVICE(int i,int j,int k)
        {
            vz_arr(i,j,k) -= (pr_arr(i,j,k)-pr_arr(i,j,k-1))/dx[2];
        });
   }
    
}
//=======================================================================
void solveManager::update_staggered_layers()
{
    const int* domlo = m_geom.Domain().loVect();
    const int* domhi = m_geom.Domain().hiVect();

    int *bclo = m_bc_lo.data();
    int *bchi = m_bc_hi.data();

    int wall     = WALL_ID;
    int inflow   = INFLOW_ID;
    int outflow  = OUTFLOW_ID;
    int periodic = PERIODIC_ID;

    for(MFIter mfi(*m_pres); mfi.isValid(); ++mfi)
    {
        const Box& bx  = mfi.validbox();
        const Box &gbx = amrex::grow(bx, m_ng);
        
        auto vx_arr=(*m_vx)[mfi].array();
        auto vy_arr=(*m_vy)[mfi].array();
        auto vz_arr=(*m_vz)[mfi].array();

        auto pr_arr=(*m_pres)[mfi].array();

        Box x_bx=convert(gbx, {1,0,0});
        Box y_bx=convert(gbx, {0,1,0});
        Box z_bx=convert(gbx, {0,0,1});

        amrex::ParallelFor(x_bx, [=] AMREX_GPU_DEVICE(int i,int j,int k)
        {       
            if(i==(domlo[0]-1) and (bclo[0]==wall || bclo[0]==inflow))
            {
                vx_arr(i+1,j,k)=vx_arr(i,j,k); 
            }
            if(i==(domhi[0]+2) and (bchi[0]==wall || bchi[0]==inflow))
            {
                vx_arr(i-1,j,k)=vx_arr(i,j,k);
            }
        });
        
        amrex::ParallelFor(y_bx, [=] AMREX_GPU_DEVICE(int i,int j,int k)
        {       
            if(j==(domlo[1]-1) and (bclo[1]==wall || bclo[1]==inflow))
            {
                vy_arr(i,j+1,k)=vy_arr(i,j,k);   
            }
            if(j==(domhi[1]+2) and (bchi[1]==wall || bchi[1]==inflow))
            {
                vy_arr(i,j-1,k)=vy_arr(i,j,k);
            }
        });
        
        amrex::ParallelFor(z_bx, [=] AMREX_GPU_DEVICE(int i,int j,int k)
        {       
            if(k==(domlo[2]-1) and (bclo[2]==wall || bclo[2]==inflow))
            {
                vz_arr(i,j,k+1)=vz_arr(i,j,k);   
            }
            if(k==(domhi[2]+2) and (bchi[2]==wall || bchi[2]==inflow))
            {
                vz_arr(i,j,k-1)=vz_arr(i,j,k);
            }
        });
   }
        
}
//=======================================================================
void solveManager::update_cc_vars()
{
    (*m_vx).FillBoundary(m_geom.periodicity());
    (*m_vy).FillBoundary(m_geom.periodicity());
    (*m_vz).FillBoundary(m_geom.periodicity());
    (*m_cc_flow_vars).FillBoundary(m_geom.periodicity());

    for(MFIter mfi(*m_cc_flow_vars);  mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.validbox();
        auto vx_arr=(*m_vx)[mfi].array();
        auto vy_arr=(*m_vy)[mfi].array();
        auto vz_arr=(*m_vz)[mfi].array();

        auto cc_arr=(*m_cc_flow_vars)[mfi].array();

        //Box x_bx=convert(bx, {1,0,0});
        //Box y_bx=convert(bx, {0,1,0});
        //Box z_bx=convert(bx, {0,0,1});
        //Print()<<"bx:"<<bx<<"\t"<<x_bx<<"\n";

        amrex::ParallelFor(bx, [=] 
                AMREX_GPU_DEVICE(int i,int j,int k)
                {
                cc_arr(i,j,k,VELX_INDX)=0.5*(vx_arr(i,j,k)+vx_arr(i+1,j,k));
                });
        amrex::ParallelFor(bx, [=] 
                AMREX_GPU_DEVICE(int i,int j,int k)
                {
                cc_arr(i,j,k,VELY_INDX)=0.5*(vy_arr(i,j,k)+vy_arr(i,j+1,k));
                });
        amrex::ParallelFor(bx, [=] 
                AMREX_GPU_DEVICE(int i,int j,int k)
                {
                cc_arr(i,j,k,VELZ_INDX)=0.5*(vz_arr(i,j,k)+vz_arr(i,j,k+1));
                });
    }
}
//=======================================================================
void solveManager::pres_poisson()
{
    LPInfo info;
    info.setMaxCoarseningLevel(100);

    const Real tol_rel = 1.e-13;
    const Real tol_abs = 0.0;

    m_pres->FillBoundary(m_geom.periodicity());

    MLPoisson mlpoisson({m_geom}, {m_ba}, {m_dmap}, info);
    mlpoisson.setMaxOrder(2);
    
    std::array<LinOpBCType,AMREX_SPACEDIM> bc_poisson_lo
        ={LinOpBCType::Neumann,LinOpBCType::Neumann,LinOpBCType::Neumann};
    std::array<LinOpBCType,AMREX_SPACEDIM> bc_poisson_hi
        ={LinOpBCType::Neumann,LinOpBCType::Neumann,LinOpBCType::Neumann};

    for(int idim=0;idim<AMREX_SPACEDIM;idim++)
    {
        if(m_bc_lo[idim] == PERIODIC_ID)
        {
            bc_poisson_lo[idim]=LinOpBCType::Periodic;
        }
        if(m_bc_lo[idim] == OUTFLOW_ID)
        {
            bc_poisson_lo[idim]=LinOpBCType::Dirichlet;
        }

        if(m_bc_hi[idim] == PERIODIC_ID)
        {
            bc_poisson_hi[idim]=LinOpBCType::Periodic;
        }
        if(m_bc_hi[idim] == OUTFLOW_ID)
        {
            bc_poisson_hi[idim]=LinOpBCType::Dirichlet;
        }
    }

   mlpoisson.setDomainBC(bc_poisson_lo,bc_poisson_hi);

   mlpoisson.setLevelBC(0, m_pres);

   MLMG mlmg(mlpoisson);
   mlmg.setMaxIter(100); //max_iter
   mlmg.setMaxFmgIter(0);
   int verbose = 2;
   mlmg.setVerbose(verbose);
   int bottom_verbose = 0;
   mlmg.setBottomVerbose(bottom_verbose);
#ifdef AMREX_USE_HYPRE
   if (use_hypre) 
   {
       mlmg.setBottomSolver(MLMG::BottomSolver::hypre);
       mlmg.setHypreInterface(hypre_interface);
   }
#endif
#ifdef AMREX_USE_PETSC
    if (use_petsc) 
    {
        mlmg.setBottomSolver(MLMG::BottomSolver::petsc);
    }
#endif
   
   mlmg.solve({m_pres}, {m_divu}, tol_rel, tol_abs);
}
//=======================================================================
void solveManager::vel_solve(Real time,Real dt)
{
    const auto dx = m_geom.CellSizeArray();

    Real visc=m_visc;
    Real delt=dt;

    for(MFIter mfi(*m_cc_flow_vars); mfi.isValid(); ++mfi)
    {
        const Box& bx  = mfi.validbox();
        
        auto vx=(*m_vx)[mfi].array();
        auto vy=(*m_vy)[mfi].array();
        auto vz=(*m_vz)[mfi].array();

        Box x_bx=convert(bx, {1,0,0});
        Box y_bx=convert(bx, {0,1,0});
        Box z_bx=convert(bx, {0,0,1});
        
        FArrayBox vx_src(x_bx, 1);
        Elixir vx_src_eli = vx_src.elixir();
        Array4<Real> const& vx_src_arr = vx_src.array();
        
        FArrayBox vy_src(y_bx, 1);
        Elixir vy_src_eli = vy_src.elixir();
        Array4<Real> const& vy_src_arr = vy_src.array();
        
        FArrayBox vz_src(z_bx, 1);
        Elixir vz_src_eli = vz_src.elixir();
        Array4<Real> const& vz_src_arr = vz_src.array();

        amrex::ParallelFor(x_bx, [=] AMREX_GPU_DEVICE(int i,int j,int k)
        {
            Real vmid;
            vx_src_arr(i,j,k)=0.0;
            //left face
            vmid=0.5*(vx(i,j,k)+vx(i-1,j,k));
            if(vmid > 0)
                vx_src_arr(i,j,k) -= vmid*vx(i-1,j,k)/dx[0];
            else
                vx_src_arr(i,j,k) -= vmid*vx(i,j,k)/dx[0];
            
            //right face
            vmid=0.5*(vx(i,j,k)+vx(i+1,j,k));
            if(vmid > 0)
                vx_src_arr(i,j,k) += vmid*vx(i,j,k)/dx[0];
            else
                vx_src_arr(i,j,k) += vmid*vx(i+1,j,k)/dx[0];
            
            //bottom face
            vmid=0.5*(vy(i-1,j,k)+vy(i,j,k));
            if(vmid > 0)
                vx_src_arr(i,j,k) -= vmid*vx(i,j-1,k)/dx[1];
            else
                vx_src_arr(i,j,k) -= vmid*vx(i,j,k)/dx[1];

            //top face
            vmid=0.5*(vy(i-1,j+1,k)+vy(i,j+1,k));
            if(vmid > 0)
                vx_src_arr(i,j,k) += vmid*vx(i,j,k)/dx[1];
            else
               vx_src_arr(i,j,k)  += vmid*vx(i,j+1,k)/dx[1];
            
            //back face
            vmid=0.5*(vz(i-1,j,k)+vz(i,j,k));
            if(vmid > 0)
                vx_src_arr(i,j,k) -= vmid*vx(i,j,k-1)/dx[2];
            else
               vx_src_arr(i,j,k)  -= vmid*vx(i,j,k)/dx[2];

            //front face
            vmid=0.5*(vz(i-1,j,k+1)+vz(i,j,k+1));
            if(vmid > 0)
                vx_src_arr(i,j,k) += vmid*vx(i,j,k)/dx[2];
            else
               vx_src_arr(i,j,k)  += vmid*vx(i,j,k+1)/dx[2];

            //diffusion
            vx_src_arr(i,j,k) -= visc*(vx(i+1,j,k)+vx(i-1,j,k)-2.0*vx(i,j,k))/(dx[0]*dx[0]);
            vx_src_arr(i,j,k) -= visc*(vx(i,j+1,k)+vx(i,j-1,k)-2.0*vx(i,j,k))/(dx[1]*dx[1]);
            vx_src_arr(i,j,k) -= visc*(vx(i,j,k+1)+vx(i,j,k-1)-2.0*vx(i,j,k))/(dx[2]*dx[2]);

            //vx(i,j,k) += -vx_src_arr(i,j,k)*delt;
        });

        amrex::ParallelFor(y_bx, [=] AMREX_GPU_DEVICE(int i,int j,int k)
        {
            Real vmid;
            vy_src_arr(i,j,k)=0.0;

            //left face
            vmid=0.5*(vx(i,j,k)+vx(i,j-1,k));
            if(vmid > 0)
                vy_src_arr(i,j,k) -= vmid*vy(i-1,j,k)/dx[0];
            else
                vy_src_arr(i,j,k) -= vmid*vy(i,j,k)/dx[0];
            
            //right face
            vmid=0.5*(vx(i+1,j-1,k)+vx(i+1,j,k));
            if(vmid > 0)
                vy_src_arr(i,j,k) += vmid*vy(i,j,k)/dx[0];
            else
                vy_src_arr(i,j,k) += vmid*vy(i+1,j,k)/dx[0];
            
            //bottom face
            vmid=0.5*(vy(i,j,k)+vy(i,j-1,k));
            if(vmid > 0)
                vy_src_arr(i,j,k) -= vmid*vy(i,j-1,k)/dx[1];
            else
               vy_src_arr(i,j,k)  -= vmid*vy(i,j,k)/dx[1];

            //top face
            vmid=0.5*(vy(i,j+1,k)+vy(i,j,k));
            if(vmid > 0)
                vy_src_arr(i,j,k) += vmid*vy(i,j,k)/dx[1];
            else
               vy_src_arr(i,j,k)  += vmid*vy(i,j+1,k)/dx[1];
            
            //back face
            vmid=0.5*(vz(i,j,k)+vz(i,j-1,k));
            if(vmid > 0)
               vy_src_arr(i,j,k) -= vmid*vy(i,j,k-1)/dx[2];
            else
               vy_src_arr(i,j,k)  -= vmid*vy(i,j,k)/dx[2];

            //front face
            vmid=0.5*(vz(i,j-1,k+1)+vz(i,j,k+1));
            if(vmid > 0)
                vy_src_arr(i,j,k) += vmid*vy(i,j,k)/dx[2];
            else
                vy_src_arr(i,j,k) += vmid*vy(i,j,k+1)/dx[2];

            //diffusion
            vy_src_arr(i,j,k) -= visc*(vy(i+1,j,k)+vy(i-1,j,k)-2.0*vy(i,j,k))/(dx[0]*dx[0]);
            vy_src_arr(i,j,k) -= visc*(vy(i,j+1,k)+vy(i,j-1,k)-2.0*vy(i,j,k))/(dx[1]*dx[1]);
            vy_src_arr(i,j,k) -= visc*(vy(i,j,k+1)+vy(i,j,k-1)-2.0*vy(i,j,k))/(dx[2]*dx[2]);
            
            //vy(i,j,k) += -vy_src_arr(i,j,k)*delt;
        });

        amrex::ParallelFor(z_bx, [=] AMREX_GPU_DEVICE(int i,int j,int k)
        {
            Real vmid;
            vz_src_arr(i,j,k)=0.0;

            //left face
            vmid=0.5*(vx(i,j,k)+vx(i,j,k-1));
            if(vmid > 0)
                vz_src_arr(i,j,k) -= vmid*vz(i-1,j,k)/dx[0];
            else
                vz_src_arr(i,j,k) -= vmid*vz(i,j,k)/dx[0];
            
            //right face
            vmid=0.5*(vx(i+1,j,k-1)+vx(i+1,j,k));
            if(vmid > 0)
                vz_src_arr(i,j,k) += vmid*vz(i,j,k)/dx[0];
            else
                vz_src_arr(i,j,k) += vmid*vz(i+1,j,k)/dx[0];
            
            //bottom face
            vmid=0.5*(vy(i,j,k)+vy(i,j,k-1));
            if(vmid > 0)
                vz_src_arr(i,j,k) -= vmid*vz(i,j-1,k)/dx[1];
            else
               vz_src_arr(i,j,k)  -= vmid*vz(i,j,k)/dx[1];

            //top face
            vmid=0.5*(vy(i,j+1,k-1)+vy(i,j,k));
            if(vmid > 0)
                vz_src_arr(i,j,k) += vmid*vz(i,j,k)/dx[1];
            else
                vz_src_arr(i,j,k) += vmid*vz(i,j+1,k)/dx[1];
            
            //back face
            vmid=0.5*(vz(i,j,k)+vz(i,j,k-1));
            if(vmid > 0)
               vz_src_arr(i,j,k) -= vmid*vz(i,j,k-1)/dx[2];
            else
               vz_src_arr(i,j,k) -= vmid*vz(i,j,k)/dx[2];

            //front face
            vmid=0.5*(vz(i,j,k)+vz(i,j,k+1));
            if(vmid > 0)
                vz_src_arr(i,j,k) += vmid*vz(i,j,k)/dx[2];
            else
                vz_src_arr(i,j,k) += vmid*vz(i,j,k+1)/dx[2];

            //diffusion
            vz_src_arr(i,j,k) -= visc*(vz(i+1,j,k)+vz(i-1,j,k)-2.0*vz(i,j,k))/(dx[0]*dx[0]);
            vz_src_arr(i,j,k) -= visc*(vz(i,j+1,k)+vz(i,j-1,k)-2.0*vz(i,j,k))/(dx[1]*dx[1]);
            vz_src_arr(i,j,k) -= visc*(vz(i,j,k+1)+vz(i,j,k-1)-2.0*vz(i,j,k))/(dx[2]*dx[2]);

            if(fabs(vz_src_arr(i,j,k)) > 0.0)
                Print()<<"vz_src:"<<vz_src_arr(i,j,k)<<"\n";

            //vz(i,j,k) += -vz_src_arr(i,j,k)*delt;
        });
        
        amrex::ParallelFor(x_bx, [=] AMREX_GPU_DEVICE(int i,int j,int k)
        {
            vx(i,j,k) += -vx_src_arr(i,j,k)*delt;
        });
        
        amrex::ParallelFor(y_bx, [=] AMREX_GPU_DEVICE(int i,int j,int k)
        {
            vy(i,j,k) += -vy_src_arr(i,j,k)*delt;
        });
        
        amrex::ParallelFor(z_bx, [=] AMREX_GPU_DEVICE(int i,int j,int k)
        {
            vz(i,j,k) += -vz_src_arr(i,j,k)*delt;
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

    m_plo.clear();
    m_phi.clear();
    m_ncells.clear();
    m_bc_lo.clear();
    m_bc_hi.clear();
    m_bcrecs.clear();
}
//=======================================================================
