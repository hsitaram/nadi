#include<solve_manager.H>
#include<global_defines.H>
#include<boundary_conditions.H>

//=======================================================================
void solveManager::set_bc_vals(Real time)
{
    const auto dx    = m_geom.CellSizeArray();
    auto prob_lo     = m_geom.ProbLoArray();

    //host pointers
    const int* domlo_hptr = m_geom.Domain().loVect();
    const int* domhi_hptr = m_geom.Domain().hiVect();
    
    GpuArray<int,AMREX_SPACEDIM> domlo={domlo_hptr[0],domlo_hptr[1],domlo_hptr[2]};
    GpuArray<int,AMREX_SPACEDIM> domhi={domhi_hptr[0],domhi_hptr[1],domhi_hptr[2]};

    GpuArray<int,AMREX_SPACEDIM> bclo={m_bc_lo[0],m_bc_lo[1],m_bc_lo[2]};
    GpuArray<int,AMREX_SPACEDIM> bchi={m_bc_hi[0],m_bc_hi[1],m_bc_hi[2]};

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
            if(i==(domlo[0]-1) and bclo[0]!=PERIODIC_ID)
            {
                apply_pres_bc(i,j,k,pr_arr,t,prob_lo,dx,XLOFACE);
            }
            if(i==(domhi[0]+1) and bchi[0]!=PERIODIC_ID)
            {
                apply_pres_bc(i,j,k,pr_arr,t,prob_lo,dx,XHIFACE);
            }
            if(j==(domlo[1]-1) and bclo[1]!=PERIODIC_ID)
            {
                apply_pres_bc(i,j,k,pr_arr,t,prob_lo,dx,YLOFACE);
            }
            if(j==(domhi[1]+1) and bchi[1]!=PERIODIC_ID)
            {
                apply_pres_bc(i,j,k,pr_arr,t,prob_lo,dx,YHIFACE);
            }
            if(k==(domlo[2]-1) and bclo[2]!=PERIODIC_ID)
            {
                apply_pres_bc(i,j,k,pr_arr,t,prob_lo,dx,ZLOFACE);
            }
            if(k==(domhi[2]+1) and bchi[2]!=PERIODIC_ID)
            {
                apply_pres_bc(i,j,k,pr_arr,t,prob_lo,dx,ZHIFACE);
            }
        });

        amrex::ParallelFor(x_bx, [=] AMREX_GPU_DEVICE(int i,int j,int k)
        {       
            if(i==(domlo[0]-1) and bclo[0]!=PERIODIC_ID)
            {
                apply_vx_bc(i,j,k,vx_arr,pr_arr,t,prob_lo,dx,XLOFACE);
            }
            if(i==(domhi[0]+2) and bchi[0]!=PERIODIC_ID)
            {
                apply_vx_bc(i,j,k,vx_arr,pr_arr,t,prob_lo,dx,XHIFACE);
            }
            if(j==(domlo[1]-1) and bclo[1]!=PERIODIC_ID)
            {
                apply_vx_bc(i,j,k,vx_arr,pr_arr,t,prob_lo,dx,YLOFACE);
            }
            if(j==(domhi[1]+1) and bchi[1]!=PERIODIC_ID)
            {
                apply_vx_bc(i,j,k,vx_arr,pr_arr,t,prob_lo,dx,YHIFACE);
            }
            if(k==(domlo[2]-1) and bclo[2]!=PERIODIC_ID)
            {
                apply_vx_bc(i,j,k,vx_arr,pr_arr,t,prob_lo,dx,ZLOFACE);
            }
            if(k==(domhi[2]+1) and bchi[2]!=PERIODIC_ID)
            {
                apply_vx_bc(i,j,k,vx_arr,pr_arr,t,prob_lo,dx,ZHIFACE);
            }
        });

        amrex::ParallelFor(y_bx, [=] 
                AMREX_GPU_DEVICE(int i,int j,int k)
        {
            if(i==(domlo[0]-1) and bclo[0]!=PERIODIC_ID)
            {
                apply_vy_bc(i,j,k,vy_arr,pr_arr,t,prob_lo,dx,XLOFACE);
            }
            if(i==(domhi[0]+1) and bchi[0]!=PERIODIC_ID)
            {
                apply_vy_bc(i,j,k,vy_arr,pr_arr,t,prob_lo,dx,XHIFACE);
            }
            if(j==(domlo[1]-1) and bclo[1]!=PERIODIC_ID)
            {
                apply_vy_bc(i,j,k,vy_arr,pr_arr,t,prob_lo,dx,YLOFACE);
            }
            if(j==(domhi[1]+2) and bchi[1]!=PERIODIC_ID)
            {
                apply_vy_bc(i,j,k,vy_arr,pr_arr,t,prob_lo,dx,YHIFACE);
            }
            if(k==(domlo[2]-1) and bclo[2]!=PERIODIC_ID)
            {
                apply_vy_bc(i,j,k,vy_arr,pr_arr,t,prob_lo,dx,ZLOFACE);
            }
            if(k==(domhi[2]+1) and bchi[2]!=PERIODIC_ID)
            {
                apply_vy_bc(i,j,k,vy_arr,pr_arr,t,prob_lo,dx,ZHIFACE);
            }
        });

        amrex::ParallelFor(z_bx, [=] 
                AMREX_GPU_DEVICE(int i,int j,int k)
        {
            if(i==(domlo[0]-1) and bclo[0]!=PERIODIC_ID)
            {
                apply_vz_bc(i,j,k,vz_arr,pr_arr,t,prob_lo,dx,XLOFACE);
            }
            if(i==(domhi[0]+1) and bchi[0]!=PERIODIC_ID)
            {
                apply_vz_bc(i,j,k,vz_arr,pr_arr,t,prob_lo,dx,XHIFACE);
            }
            if(j==(domlo[1]-1) and bclo[1]!=PERIODIC_ID)
            {
                apply_vz_bc(i,j,k,vz_arr,pr_arr,t,prob_lo,dx,YLOFACE);
            }
            if(j==(domhi[1]+1) and bchi[1]!=PERIODIC_ID)
            {
                apply_vz_bc(i,j,k,vz_arr,pr_arr,t,prob_lo,dx,YHIFACE);
            }
            if(k==(domlo[2]-1) and bclo[2]!=PERIODIC_ID)
            {
                apply_vz_bc(i,j,k,vz_arr,pr_arr,t,prob_lo,dx,ZLOFACE);
            }
            if(k==(domhi[2]+2) and bchi[2]!=PERIODIC_ID)
            {
                apply_vz_bc(i,j,k,vz_arr,pr_arr,t,prob_lo,dx,ZHIFACE);
            }
        });

   }
}
//=======================================================================
void solveManager::update_staggered_layers()
{
    //Host pointers
    const int* domlo_hptr = m_geom.Domain().loVect();
    const int* domhi_hptr = m_geom.Domain().hiVect();
    
    GpuArray<int,AMREX_SPACEDIM> domlo={domlo_hptr[0],domlo_hptr[1],domlo_hptr[2]};
    GpuArray<int,AMREX_SPACEDIM> domhi={domhi_hptr[0],domhi_hptr[1],domhi_hptr[2]};

    GpuArray<int,AMREX_SPACEDIM> bclo={m_bc_lo[0],m_bc_lo[1],m_bc_lo[2]};
    GpuArray<int,AMREX_SPACEDIM> bchi={m_bc_hi[0],m_bc_hi[1],m_bc_hi[2]};

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
            if(i==(domlo[0]-1) and (bclo[0]==WALL_ID || bclo[0]==INFLOW_ID))
            {
                vx_arr(i+1,j,k)=vx_arr(i,j,k); 
            }
            if(i==(domhi[0]+2) and (bchi[0]==WALL_ID || bchi[0]==INFLOW_ID))
            {
                vx_arr(i-1,j,k)=vx_arr(i,j,k);
            }
        });
        
        amrex::ParallelFor(y_bx, [=] AMREX_GPU_DEVICE(int i,int j,int k)
        {       
            if(j==(domlo[1]-1) and (bclo[1]==WALL_ID || bclo[1]==INFLOW_ID))
            {
                vy_arr(i,j+1,k)=vy_arr(i,j,k);   
            }
            if(j==(domhi[1]+2) and (bchi[1]==WALL_ID || bchi[1]==INFLOW_ID))
            {
                vy_arr(i,j-1,k)=vy_arr(i,j,k);
            }
        });
        
        amrex::ParallelFor(z_bx, [=] AMREX_GPU_DEVICE(int i,int j,int k)
        {       
            if(k==(domlo[2]-1) and (bclo[2]==WALL_ID || bclo[2]==INFLOW_ID))
            {
                vz_arr(i,j,k+1)=vz_arr(i,j,k);   
            }
            if(k==(domhi[2]+2) and (bchi[2]==WALL_ID || bchi[2]==INFLOW_ID))
            {
                vz_arr(i,j,k-1)=vz_arr(i,j,k);
            }
        });
   }
        
}
//=======================================================================
