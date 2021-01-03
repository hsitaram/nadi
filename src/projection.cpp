#include<solve_manager.H>
#include<global_defines.H>
#include <AMReX_MLMG.H>
#include <AMReX_MLPoisson.H>

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
