#include<solve_manager.H>
#include<global_defines.H>

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
