#include<solve_manager.H>
#include<global_defines.H>
#include<boundary_conditions.H>

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
