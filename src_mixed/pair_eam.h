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

#ifndef PAIR_EAM_H
#define PAIR_EAM_H

#include "pair.h"

namespace LAMMPS_NS {

class PairEAM : public Pair {
 public:
  double cutforcesq,cutmax;
  int n_step;
  // per-atom arrays

  double *rho,*fp;
  double *up_ADP,*wp_ADP; //u', w'
	//Additional ADP functions
//  double *mu_x,*mu_y,*mu_z;
  double *lambda_xx,*lambda_yy,*lambda_zz,*lambda_xy,*lambda_yz,*lambda_xz;  /* ADP functions */
// double *e_pair;  //for printing pair potential at an atom
  // potentials as array data

  int nrho,nr;
  int nfrho,nrhor,nz2r;
  double **frho,**rhor,**z2r;
  int *type2frho,**type2rhor,**type2z2r;
	//Additional ADP variables
  int nu_ADP,nw_ADP;
  double **u_ADP,**w_ADP;
  int **type2u_ADP,**type2w_ADP;
  // potentials in spline form used for force computation

  double dr,rdr,drho,rdrho;
  double ***rhor_spline,***frho_spline,***z2r_spline;
	//Additional ADP variables
  double ***u_ADP_spline,***w_ADP_spline;
  
  PairEAM(class LAMMPS *);
  virtual ~PairEAM();
  void compute(int, int);
  void settings(int, char **);
  virtual void coeff(int, char **);
  void init_style();
  double init_one(int, int);
  double single(int, int, int, int, double, double, double, double &);

  int pack_comm(int, int *, double *, int, int *);
  void unpack_comm(int, int, double *);
  int pack_reverse_comm(int, int, double *);
  void unpack_reverse_comm(int, int *, double *);
  double memory_usage();
  void swap_eam(double *, double **);

 protected:
  int nmax;                   // allocated size of per-atom arrays

  // potentials as file data

  int *map;                   // which element each atom type maps to

  struct Funcfl {
    char *file;
    int nrho,nr;
    double drho,dr,cut,mass;
    double *frho,*rhor,*zr;
	double *u_ADP,*w_ADP;  // Additional for ADP
  };
  Funcfl *funcfl;
  int nfuncfl;

  struct Setfl {
    char **elements;
    int nelements,nrho,nr;
    double drho,dr,cut;
    double *mass;
    double **frho,**rhor,***z2r;
    double ***u_ADP,***w_ADP;  // Additional for ADP
  };
  Setfl *setfl;

  struct Fs {
    char **elements;
    int nelements,nrho,nr;
    double drho,dr,cut;
    double *mass;
    double **frho,***rhor,***z2r;
  };
  Fs *fs;

  void allocate();
  void array2spline();
  void interpolate(int, double, double *, double **);
  void grab(FILE *, int, double *);

  virtual void read_file(char *);
  virtual void file2array();
};

}

#endif
