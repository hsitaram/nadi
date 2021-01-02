#ifndef _EXTERNBC_H_
#define _EXTERNBC_H_

#include <AMReX_Box.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>
#include <globalDefines.H>

using namespace amrex;

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void externalbc(const amrex::Real x[AMREX_SPACEDIM],
        const amrex::Real s_int[],
        amrex::Real s_ext[],
        const int idir,
        const int sgn,
        const amrex::Real time,
        amrex::GeometryData const& geomdata)
{
    s_ext[PRES_INDX] = s_int[PRES_INDX];
    s_ext[VELX_INDX] = s_int[VELX_INDX];
    s_ext[VELY_INDX] = s_int[VELY_INDX];
    s_ext[VELZ_INDX] = s_int[VELZ_INDX];
}
#endif
