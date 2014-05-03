/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifndef FIX_NVE_H
#define FIX_NVE_H

#include "fix.h"
#include "pointers.h"

namespace LAMMPS_NS {

class FixNVE : public Fix {
 public:
  FixNVE(class LAMMPS *, int, char **);
  int setmask();
  virtual void init();
  virtual void initial_integrate(int);
  virtual void final_integrate();
  void initial_integrate_respa(int, int, int);
  void final_integrate_respa(int);
  void reset_dt();
 private:
  class ComputeTemp *temper;
  void zero_momentum();
  void zero_rotation();
  void rescale_vel_temp(double t_old, double t_new);
  int i4_uniform_ab ( int a, int b, int *seed );
  double*** calloc_2d(long int l,long int m, long int n); 
 protected:
  double dtv,dtf;
  double *step_respa;
  int mass_require;
};

}

#endif
