/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   www.cs.sandia.gov/~sjplimp/lammps.html
   Steve Plimpton, sjplimp@sandia.gov, Sandia National Laboratories

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifndef ATOM_VEC_DIPOLE_H
#define ATOM_VEC_DIPOLE_H

#include "atom_vec.h"

namespace LAMMPS_NS {

class AtomVecDipole : public AtomVec {
 public:
  AtomVecDipole(class LAMMPS *, int, char **);
  void grow(int);
  void copy(int, int);
  int pack_comm(int, int *, double *, int, int *);
  int pack_comm_one(int, double *);
  void unpack_comm(int, int, double *);
  int unpack_comm_one(int, double *);
  int pack_reverse(int, int, double *);
  int pack_reverse_one(int, double *);
  void unpack_reverse(int, int *, double *);
  int unpack_reverse_one(int, double *);
  int pack_border(int, int *, double *, int, int *);
  int pack_border_one(int, double *);
  void unpack_border(int, int, double *);
  int unpack_border_one(int, double *);
  int pack_exchange(int, double *);
  int unpack_exchange(double *);
  int size_restart();
  int pack_restart(int, double *);
  int unpack_restart(double *);
  void create_atom(int, double *);
  void data_atom(double *, int, char **);
  int data_atom_hybrid(int, char **);
  void data_vel(int, char **);
  int data_vel_hybrid(int, char **);
  double memory_usage();

 private:
  int *tag,*type,*mask,*image;
  double **x,**v,**f;
  double *q,**mu,**omega,**torque;
};

}

#endif
