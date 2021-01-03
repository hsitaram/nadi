#ifndef BOUNDARY_CONDITIONS_H_
#define BOUNDARY_CONDITIONS_H_
#include <global_defines.H>

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void apply_pres_bc(const int i,
        const int j,const int k,
        Array4<Real>const& pres,
        Real time,
        GpuArray<Real, AMREX_SPACEDIM> prob_lo,
        GpuArray<Real, AMREX_SPACEDIM> dx,
        const int face)
{

    if(face==XLOFACE)
    {
        pres(i,j,k)=pres(i+1,j,k); //zero gradient
    }
    else if(face==XHIFACE)
    {
        pres(i,j,k)=pres(i-1,j,k); //zero gradient
    }
    else if(face==YLOFACE)
    {
        pres(i,j,k)=pres(i,j+1,k); //zero gradient
    }
    else if(face==YHIFACE)
    {
        pres(i,j,k)=pres(i,j-1,k); //zero gradient
    }
    else if(face==ZLOFACE)
    {
        pres(i,j,k)=pres(i,j,k+1); //zero gradient
    }
    else
    {
        pres(i,j,k)=pres(i,j,k-1); //zero gradient
    }
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void apply_vx_bc(const int i,
        const int j,const int k,
        Array4<Real>const& vx,
        Array4<Real>const& pres,
        Real time,
        GpuArray<Real, AMREX_SPACEDIM> prob_lo,
        GpuArray<Real, AMREX_SPACEDIM> dx,
        const int face)
{
    if(face==XLOFACE) //WALL
    {
        vx(i,j,k)=0.0;
    }
    else if(face==XHIFACE)
    {
        vx(i,j,k)=0.0;
    }
    else if(face==YLOFACE)
    {
        vx(i,j,k)=-vx(i,j+1,k); //reflect odd
    }
    else if(face==YHIFACE)
    {
        vx(i,j,k)=2.0*1.0-vx(i,j-1,k);
    }
    else if(face==ZLOFACE)
    {
        vx(i,j,k)=-vx(i,j,k+1);
    }
    else
    {
        vx(i,j,k)=-vx(i,j,k-1);
    }
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void apply_vy_bc(const int i,
        const int j,const int k,
        Array4<Real>const& vy,
        Array4<Real>const& pres,
        Real time,
        GpuArray<Real, AMREX_SPACEDIM> prob_lo,
        GpuArray<Real, AMREX_SPACEDIM> dx,
        const int face)
{
    if(face==XLOFACE)
    {
        vy(i,j,k)=-vy(i+1,j,k); //reflect odd
    }
    else if(face==XHIFACE)
    {
        vy(i,j,k)=-vy(i-1,j,k);
    }
    else if(face==YLOFACE)
    {
        vy(i,j,k)=0.0;
    }
    else if(face==YHIFACE)
    {
        vy(i,j,k)=0.0;
    }
    else if(face==ZLOFACE)
    {
        vy(i,j,k)=-vy(i,j,k+1);
    }
    else
    {
        vy(i,j,k)=-vy(i,j,k-1);
    }
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void apply_vz_bc(const int i,
        const int j,const int k,
        Array4<Real>const& vz,
        Array4<Real>const& pres,
        Real time,
        GpuArray<Real, AMREX_SPACEDIM> prob_lo,
        GpuArray<Real, AMREX_SPACEDIM> dx,
        const int face)
{
    if(face==XLOFACE)
    {
        vz(i,j,k)=0.0;
    }
    else if(face==XHIFACE)
    {
        vz(i,j,k)=0.0;
    }
    else if(face==YLOFACE)
    {
        vz(i,j,k)=0.0;
    }
    else if(face==YHIFACE)
    {
        vz(i,j,k)=0.0;
    }
    else if(face==ZLOFACE)
    {
        vz(i,j,k)=0.0;
    }
    else
    {
        vz(i,j,k)=0.0;
    }
}
#endif
