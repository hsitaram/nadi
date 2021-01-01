#ifndef INITIALIZE_FIELDS_H_
#define INITIALIZE_FIELDS_H_

AMREX_GPU_DEVICE AMREX_INLINE
void set_vx_ic(int i, int j, int k,
        Array4<Real> const& vx,
        GpuArray<Real, AMREX_SPACEDIM> prob_lo,
        GpuArray<Real, AMREX_SPACEDIM> dx)
{
   vx(i,j,k) = 0.0;
}

AMREX_GPU_DEVICE AMREX_INLINE
void set_vy_ic(int i, int j, int k,
        Array4<Real> const& vy,
        GpuArray<Real, AMREX_SPACEDIM> prob_lo,
        GpuArray<Real, AMREX_SPACEDIM> dx)
{
   vy(i,j,k) = 0.0;
}

AMREX_GPU_DEVICE AMREX_INLINE
void set_vz_ic(int i, int j, int k,
        Array4<Real> const& vz,
        GpuArray<Real, AMREX_SPACEDIM> prob_lo,
        GpuArray<Real, AMREX_SPACEDIM> dx)
{
   vz(i,j,k) = 0.0;
}

AMREX_GPU_DEVICE AMREX_INLINE
void set_pres_ic(int i, int j, int k,
        Array4<Real> const& pres,
        GpuArray<Real, AMREX_SPACEDIM> prob_lo,
        GpuArray<Real, AMREX_SPACEDIM> dx)
{
   pres(i,j,k) = 0.0;
}

#endif
