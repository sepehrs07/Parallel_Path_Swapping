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

/* ----------------------------------------------------------------------
   Contributing authors: Stephen Foiles (SNL), Murray Daw (SNL)
------------------------------------------------------------------------- */

/*
------------------------------------------------------------
Implementation of Mishin's Angular Dependent Potential (ADP)
By: Chandra Veer Singh, Cornell University
---------------------------------------------------------- */

/*************************************************************
                Definitions of parameters
**************************************************************
  rho = density
  frho = embedding energy
  rhor = sum of density at an atom
  z2r = Zi*Zj
  phi = pair potential energy
  fp = derivative of embedding energy
  u_ADP = dipole potential function for ADP
  w_ADP = quadrupole potential function for ADP
  up_ADP = derivative of u_ADP w.r.t. r
  wp_ADP = derivative of w_ADP w.r.t. r
  rhoip = derivative of (density at atom j due to atom i)
  rhojp = derivative of (density at atom i due to atom j)
  phip = phi'
  z2 = phi * r = 27.2 * 0.529 * Zi * Zj
  z2p = (phi * r)' = (phi' r) + phi
  mu_ADP = dipole potential vector
  lambda_ADP = quadrupole tensor
  v_ADP = Trace(lambda_ADP)
***************************************************************/

#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "pair_eam.h"
#include "atom.h"
#include "force.h"
#include "comm.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "memory.h"
#include "error.h"
#include "vector" 

using namespace std;
using namespace LAMMPS_NS;

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

#define MAXLINE 1024

/* ---------------------------------------------------------------------- */

PairEAM::PairEAM(LAMMPS *lmp) : Pair(lmp)
{
  nmax = 0;
  rho = NULL;
  fp = NULL;

  nfuncfl = 0;
  funcfl = NULL;

  setfl = NULL;
  fs = NULL;

  frho = NULL;
  rhor = NULL;
  z2r = NULL;
  u_ADP = NULL;
  w_ADP = NULL;

  frho_spline = NULL;
  rhor_spline = NULL;
  z2r_spline = NULL;
  u_ADP_spline = NULL;
  w_ADP_spline = NULL;

// set comm size needed by this Pair
  comm_forward = 1;
  comm_reverse = 1;
}

/* ----------------------------------------------------------------------
   check if allocated, since class can be destructed when incomplete
------------------------------------------------------------------------- */

PairEAM::~PairEAM()
{
  memory->sfree(rho);
  memory->sfree(fp);

  if (allocated) {
    memory->destroy_2d_int_array(setflag);
    memory->destroy_2d_double_array(cutsq);
    delete [] map;
    delete [] type2frho;
    memory->destroy_2d_int_array(type2rhor);
    memory->destroy_2d_int_array(type2z2r);
    memory->destroy_2d_int_array(type2u_ADP);			
    memory->destroy_2d_int_array(type2w_ADP);			
  }

  if (funcfl) {
    for (int i = 0; i < nfuncfl; i++) {
      delete [] funcfl[i].file;
      memory->sfree(funcfl[i].frho);
      memory->sfree(funcfl[i].rhor);
      memory->sfree(funcfl[i].zr);
      
    }
    memory->sfree(funcfl);
  }

  if (setfl) {
    for (int i = 0; i < setfl->nelements; i++) delete [] setfl->elements[i];
    delete [] setfl->elements;
    delete [] setfl->mass;
    memory->destroy_2d_double_array(setfl->frho);
    memory->destroy_2d_double_array(setfl->rhor);
    memory->destroy_3d_double_array(setfl->z2r);
    memory->destroy_3d_double_array(setfl->u_ADP);	/*   mu_ADP = u_ADP*rij           */
    memory->destroy_3d_double_array(setfl->w_ADP);	/*   lambda_ADP = w_ADP*rij*rij   */
    delete setfl;
  }

  if (fs) {
    for (int i = 0; i < fs->nelements; i++) delete [] fs->elements[i];
    delete [] fs->elements;
    delete [] fs->mass;
    memory->destroy_2d_double_array(fs->frho);
    memory->destroy_3d_double_array(fs->rhor);
    memory->destroy_3d_double_array(fs->z2r);
    delete fs;
  }

  memory->destroy_2d_double_array(frho);
  memory->destroy_2d_double_array(rhor);
  memory->destroy_2d_double_array(z2r);
  memory->destroy_2d_double_array(u_ADP);
  memory->destroy_2d_double_array(w_ADP);

  memory->destroy_3d_double_array(frho_spline);
  memory->destroy_3d_double_array(rhor_spline);
  memory->destroy_3d_double_array(z2r_spline);
  memory->destroy_3d_double_array(u_ADP_spline);
  memory->destroy_3d_double_array(w_ADP_spline);
}

/* ---------------------------------------------------------------------- */

void PairEAM::compute(int eflag, int vflag)
{
  int i,j,ii,jj,m,inum,jnum,itype,jtype;
  double xtmp,ytmp,ztmp,delx,dely,delz,evdwl,fpair;
  double rsq,r,p,rhoip,rhojp,z2,z2p,recip,phip,psip,phi;				 // for EAM force contribution
  double *coeff;
  int *ilist,*jlist,*numneigh,**firstneigh;
	//Additional variables for ADP
  double tmp,pot,tr;
  double u_ADP,w_ADP,v_ADP,up_ADP,wp_ADP;  /* ADP functions */
  double dmu_ADPx,dmu_ADPy,dmu_ADPz,la_xx,la_yy,la_zz,la_xy,la_xz,la_yz; // for ADP force contribution  
  double v_x,v_y,v_z,nu,f1,f2;
  double sprod_mud,sprod_vd;

  evdwl = 0.0;
  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = vflag_fdotr = 0;

  // grow energy and fp arrays if necessary
  // need to be atom->nmax in length

  if (atom->nmax > nmax) {
    memory->sfree(rho);
    memory->sfree(fp);
    nmax = atom->nmax;
    rho = (double *) memory->smalloc(nmax*sizeof(double),"pair:rho");
    fp = (double *) memory->smalloc(nmax*sizeof(double),"pair:fp");
  }

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int newton_pair = force->newton_pair;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // zero out density

  if (newton_pair) {                                  // Newton pair turns Newton's 3rd law on or off for atomic/bonded interactions.
    m = nlocal + atom->nghost;
    for (i = 0; i < m; i++) rho[i] = 0.0;
  } else for (i = 0; i < nlocal; i++) rho[i] = 0.0;

  // rho = density at each atom
  // loop over neighbors of my atoms

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;

      if (rsq < cutforcesq) {
	jtype = type[j];
	p = sqrt(rsq)*rdr + 1.0;
	m = static_cast<int> (p);
	m = MIN(m,nr-1);
	p -= m;
	p = MIN(p,1.0);
	coeff = rhor_spline[type2rhor[jtype][itype]][m];
	rho[i] += ((coeff[3]*p + coeff[4])*p + coeff[5])*p + coeff[6];
	
	if (newton_pair || j < nlocal) {
	  coeff = rhor_spline[type2rhor[itype][jtype]][m];
	  rho[j] += ((coeff[3]*p + coeff[4])*p + coeff[5])*p + coeff[6];
	}
      }
    }
  }
  // communicate and sum densities

  if (newton_pair) comm->reverse_comm_pair(this);

/*******************************************************
******* Calculation of additional ADP functions  *******
*******************************************************/
// Calculate mu_ADP, lambda_ADP and u', w';             u' and w' are still to be implemented
  static vector<double> mu_x;
  static vector<double> mu_y;
  static vector<double> mu_z;
  static vector<double> lambda_xx;
  static vector<double> lambda_yy;
  static vector<double> lambda_zz;
  static vector<double> lambda_xy;
  static vector<double> lambda_xz;
  static vector<double> lambda_yz;
  static int dwcounter=0;
  dwcounter++;
  if(dwcounter==1){
   mu_x.resize(nmax);
   mu_y.resize(nmax);
   mu_z.resize(nmax);
   lambda_xx.resize(nmax);
   lambda_yy.resize(nmax);
   lambda_zz.resize(nmax);
   lambda_xy.resize(nmax);
   lambda_xz.resize(nmax);
   lambda_yz.resize(nmax);
  } 
// Initialize mu and lambda terms
for (ii = 0; ii < inum; ii++) { 
  i = ilist[ii]; 
  mu_x[i]=mu_y[i]=mu_z[i]=0.0;
  lambda_xx[i]= lambda_yy[i]= lambda_zz[i]= lambda_xy[i]= lambda_xz[i]= lambda_yz[i]=0.0;
}
// calculate mu and lambda terms
//if (itype != jtype){  // Avoid ADP calculation for same element interactions
for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];

    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;

      if (rsq < cutforcesq) {		
	jtype = type[j];

	r = sqrt(rsq);
	
	p = r*rdr + 1.0;
	m = static_cast<int> (p);
	m = MIN(m,nr-1);
	p -= m;
	p = MIN(p,1.0);
// mu_ADP
	coeff = u_ADP_spline[type2u_ADP[itype][jtype]][m];
	u_ADP = ((coeff[3]*p + coeff[4])*p + coeff[5])*p + coeff[6];

	mu_x[i] += u_ADP * delx; // mu_x
    	mu_y[i] += u_ADP * dely; // mu_y
    	mu_z[i] += u_ADP * delz; // mu_z

	if (newton_pair || j < nlocal) {
          mu_x[j] -= u_ADP * delx; // mu_x
          mu_y[j] -= u_ADP * dely; // mu_y
          mu_z[j] -= u_ADP * delz; // mu_z
	}
// lambda_ADP
	coeff = w_ADP_spline[type2w_ADP[itype][jtype]][m];
	w_ADP = ((coeff[3]*p + coeff[4])*p + coeff[5])*p + coeff[6];
	
	lambda_xx[i] += w_ADP * delx * delx; //lambda_xx
    	lambda_yy[i] += w_ADP * dely * dely; //lambda_yy
    	lambda_zz[i] += w_ADP * delz * delz; //lambda_zz
    	lambda_yz[i] += w_ADP * dely * delz; //lambda_yz
    	lambda_xz[i] += w_ADP * delx * delz; //lambda_xz
    	lambda_xy[i] += w_ADP * delx * dely; //lambda_xy
        
	if (newton_pair || j < nlocal) {
         lambda_xx[j] += w_ADP * delx * delx; //lambda_xx
         lambda_yy[j] += w_ADP * dely * dely; //lambda_yy
         lambda_zz[j] += w_ADP * delz * delz; //lambda_zz
         lambda_yz[j] += w_ADP * dely * delz; //lambda_yz
         lambda_xz[j] += w_ADP * delx * delz; //lambda_xz
         lambda_xy[j] += w_ADP * delx * dely; //lambda_xy
        }

     }
   }
}
//} // End loop for itype !=jtype if statement
/*******************************************************
************  Calculation of  potential  ***************
*******************************************************/
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    p = rho[i]*rdrho + 1.0;
    m = static_cast<int> (p);
    m = MAX(1,MIN(m,nrho-1));
    p -= m;
    p = MIN(p,1.0);
    coeff = frho_spline[type2frho[type[i]]][m];
    fp[i] = (coeff[0]*p + coeff[1])*p + coeff[2];
    
	if (eflag) {
      	pot = ((coeff[3]*p + coeff[4])*p + coeff[5])*p + coeff[6]; // Frho interpolation
      	
	tmp = mu_x[i]; pot += 0.5*tmp*tmp;
	tmp = mu_y[i]; pot += 0.5*tmp*tmp;
	tmp = mu_z[i]; pot += 0.5*tmp*tmp;
	tr  = (lambda_xx[i]+lambda_yy[i]+lambda_zz[i]); // v_ADP = Trace(lambda_ADP)
	pot += - tr*tr/6.0;
	tmp = lambda_xx[i];	pot += 0.5*tmp*tmp;
      	tmp = lambda_yy[i];	pot += 0.5*tmp*tmp;
    	tmp = lambda_zz[i];	pot += 0.5*tmp*tmp;
  	tmp = lambda_yz[i];	pot += tmp*tmp;
      	tmp = lambda_xz[i];	pot += tmp*tmp;
      	tmp = lambda_xy[i];	pot += tmp*tmp;

	if (eflag_global) eng_vdwl += pot;
     	if (eflag_atom) eatom[i] += pot;
    }
  }
 // communicate derivative of embedding function
  comm->comm_pair(this);

/*******************************************************
***************  Force Calculation  ********************
*******************************************************/
double fx_eam, fy_eam, fz_eam;
double fx_dpot, fy_dpot, fz_dpot, fx_qpot, fy_qpot, fz_qpot;
fx_eam = fy_eam = fz_eam = 0;
fx_dpot = fy_dpot = fz_dpot = fx_qpot = fy_qpot = fz_qpot =0;
// compute forces on each atom
  // loop over neighbors of my atoms

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;

      if (rsq < cutforcesq) {
	jtype = type[j];
	r = sqrt(rsq);
	p = r*rdr + 1.0;
	m = static_cast<int> (p);
	m = MIN(m,nr-1);
	p -= m;
	p = MIN(p,1.0);

	// rhoip = derivative of (density at atom j due to atom i)
	// rhojp = derivative of (density at atom i due to atom j)
	// phi = pair potential energy
	// phip = phi', z2 = phi*r
	// z2p=(phi*r)'=(phi'r)+phi ==> phip=1/r*{z2p-phi}=1/r*{(phi'*r+phi)-phi}=phi'
	// v_ADP = Trace(lambda_ADP)
	// u_ADP = dipole potential function
	// w_ADP = quadrupole potential function
	// up_ADP = derivative of dipole potential function
	// wp_ADP = derivative of quadrupole potential function
	// psip needs both fp[i] and fp[j] terms since r_ij appears in two
	//   terms of embed eng: Fi(sum rho_ij) and Fj(sum rho_ji)
	//   hence embed' = Fi(sum rho_ij) rhojp + Fj(sum rho_ji) rhoip

	coeff = rhor_spline[type2rhor[itype][jtype]][m];
	rhoip = (coeff[0]*p + coeff[1])*p + coeff[2];
	coeff = rhor_spline[type2rhor[jtype][itype]][m];
	rhojp = (coeff[0]*p + coeff[1])*p + coeff[2];
	coeff = z2r_spline[type2z2r[itype][jtype]][m];
	z2p = (coeff[0]*p + coeff[1])*p + coeff[2];
	z2 = ((coeff[3]*p + coeff[4])*p + coeff[5])*p + coeff[6];

	recip = 1.0/r;
	phi = z2*recip;
	phip = z2p*recip - phi*recip;
	psip = fp[i]*rhojp + fp[j]*rhoip + phip;
	fpair = -psip*recip;
	
//Forces due to EAM contribution
	fx_eam = delx*fpair;
	fy_eam = dely*fpair;
	fz_eam = delz*fpair;
  
// Force terms from ADP potential //
//if (itype != jtype){
    	/* forces due to dipole distortion */
          dmu_ADPx = mu_x[i] - mu_x[j];
          dmu_ADPy = mu_y[i] - mu_y[j];
          dmu_ADPz = mu_z[i] - mu_z[j];
          coeff = u_ADP_spline[type2u_ADP[itype][jtype]][m];
	  up_ADP = (coeff[0]*p + coeff[1])*p + coeff[2];
	  pot = ((coeff[3]*p + coeff[4])*p + coeff[5])*p + coeff[6];
	  sprod_mud = recip * (dmu_ADPx * delx + dmu_ADPy * dely + dmu_ADPz * delz); //SPROD(mu,d), Scalar product of dmu & d
	  tmp  = sprod_mud * up_ADP;  // get u'=up_ADP 
          fx_dpot = - dmu_ADPx * pot - tmp * delx;
          fy_dpot = - dmu_ADPy * pot - tmp * dely;
          fz_dpot = - dmu_ADPz * pot - tmp * delz;
         
        /* forces due to quadrupole distortion */
          la_xx = lambda_xx[i]+lambda_xx[j];
          la_yy = lambda_yy[i]+lambda_yy[j];
          la_zz = lambda_zz[i]+lambda_zz[j];
          la_yz = lambda_yz[i]+lambda_yz[j];
          la_xz = lambda_xz[i]+lambda_xz[j];
          la_xy = lambda_xy[i]+lambda_xy[j];
          v_x = la_xx * delx + la_xy * dely + la_xz * delz;
          v_y = la_xy * delx + la_yy * dely + la_yz * delz;
          v_z = la_xz * delx + la_yz * dely + la_zz * delz;
          nu  = (la_xx + la_yy + la_zz) / 3.0;
          coeff = w_ADP_spline[type2w_ADP[itype][jtype]][m];
	  wp_ADP = (coeff[0]*p + coeff[1])*p + coeff[2];
	  pot = ((coeff[3]*p + coeff[4])*p + coeff[5])*p + coeff[6];
	  //from IMD code: r2=SPROD(d,d)=rsq; SPROD is scalar (dot) product,
	  // f1  = 2.0 * pot;f2  = (SPROD(v,d) - nu * r2) * grad - nu * f1; grad is
          sprod_vd= recip * (v_x * delx + v_y * dely + v_z * delz); //SPROD(v,d), Scalar product of v & d
	  f2  = (sprod_vd - nu * r) * wp_ADP - nu * 2.0 * pot; // get wp_ADP=w', r2=?, SPROD? 
          fx_qpot = - 2.0 * pot * v_x - f2 * delx;
          fy_qpot = - 2.0 * pot * v_y - f2 * dely;
          fz_qpot = - 2.0 * pot * v_z - f2 * delz;
//   } // End loop for itype != jtype if statement
       // Total forces = EAM + ADP contribution
	  f[i][0] += fx_eam + fx_dpot + fx_qpot;
	  f[i][1] += fy_eam + fy_dpot + fy_qpot;
	  f[i][2] += fz_eam + fz_dpot + fz_qpot;

	if (newton_pair || j < nlocal) {
	  f[j][0] -= fx_eam + fx_dpot + fx_qpot;
	  f[j][1] -= fy_eam + fy_dpot + fy_qpot;
	  f[j][2] -= fz_eam + fz_dpot + fz_qpot;
	}
	if (eflag) evdwl = phi;		//evwl is to tally energy with virial algorithm.
	if (evflag) ev_tally(i,j,nlocal,newton_pair,
			     evdwl,0.0,fpair,delx,dely,delz);
	}
   }
}
  if (vflag_fdotr) virial_compute();

}  // End of PairEAM::Compute 

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */
void PairEAM::allocate()
{
  allocated = 1;
  int n = atom->ntypes;
  setflag = memory->create_2d_int_array(n+1,n+1,"pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  cutsq = memory->create_2d_double_array(n+1,n+1,"pair:cutsq");

  map = new int[n+1];
  for (int i = 1; i <= n; i++) map[i] = -1;

  type2frho = new int[n+1];
  type2rhor = memory->create_2d_int_array(n+1,n+1,"pair:type2rhor");
  type2z2r = memory->create_2d_int_array(n+1,n+1,"pair:type2z2r");
  type2u_ADP = memory->create_2d_int_array(n+1,n+1,"pair:type2u_ADP");
  type2w_ADP = memory->create_2d_int_array(n+1,n+1,"pair:type2w_ADP");
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairEAM::settings(int narg, char **arg)
{
  if (narg > 0) error->all("Illegal pair_style command");
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs		
   read DYNAMO funcfl file
------------------------------------------------------------------------- */

void PairEAM::coeff(int narg, char **arg)
{
  if (!allocated) allocate();

  if (narg != 3) error->all("Incorrect args for pair coefficients");

  // parse pair of atom types

  int ilo,ihi,jlo,jhi;
  force->bounds(arg[0],atom->ntypes,ilo,ihi);
  force->bounds(arg[1],atom->ntypes,jlo,jhi);

  // read funcfl file if hasn't already been read
  // store filename in Funcfl data struct

  int ifuncfl;
  for (ifuncfl = 0; ifuncfl < nfuncfl; ifuncfl++)
    if (strcmp(arg[2],funcfl[ifuncfl].file) == 0) break;

  if (ifuncfl == nfuncfl) {
    nfuncfl++;
    funcfl = (Funcfl *) 
    memory->srealloc(funcfl,nfuncfl*sizeof(Funcfl),"pair:funcfl");
    read_file(arg[2]);
    int n = strlen(arg[2]) + 1;
    funcfl[ifuncfl].file = new char[n];
    strcpy(funcfl[ifuncfl].file,arg[2]);
  }

  // set setflag and map only for i,i type pairs
  // set mass of atom type if i = j

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      if (i == j) {
	setflag[i][i] = 1;
	map[i] = ifuncfl;
	atom->set_mass(i,funcfl[ifuncfl].mass);
	count++;
      }
    }
  }

  if (count == 0) error->all("Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairEAM::init_style()
{
  // convert read-in file(s) to arrays and spline them

  file2array();
  array2spline();

  int irequest = neighbor->request(this);
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairEAM::init_one(int i, int j)
{
  // single global cutoff = max of cut from all files read in
  // for funcfl could be multiple files
  // for setfl or fs, just one file

  if (funcfl) {
    cutmax = 0.0;
    for (int m = 0; m < nfuncfl; m++)
      cutmax = MAX(cutmax,funcfl[m].cut);
  } else if (setfl) cutmax = setfl->cut;
  else if (fs) cutmax = fs->cut;

  cutforcesq = cutmax*cutmax;

  return cutmax;
}

/* ----------------------------------------------------------------------
   read potential values from a DYNAMO single element funcfl file 
------------------------------------------------------------------------- */

void PairEAM::read_file(char *filename)
{
  Funcfl *file = &funcfl[nfuncfl-1];

  int me = comm->me;
  FILE *fp;
  char line[MAXLINE];

  if (me == 0) {
    fp = fopen(filename,"r");
    if (fp == NULL) {
      char str[128];
      sprintf(str,"Cannot open EAM potential file %s",filename);
      error->one(str);
    }
  }

  int tmp;
  if (me == 0) {
    fgets(line,MAXLINE,fp);
    fgets(line,MAXLINE,fp);
    sscanf(line,"%d %lg",&tmp,&file->mass);
    fgets(line,MAXLINE,fp);
    sscanf(line,"%d %lg %d %lg %lg",
	   &file->nrho,&file->drho,&file->nr,&file->dr,&file->cut);
  }

  MPI_Bcast(&file->mass,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&file->nrho,1,MPI_INT,0,world);
  MPI_Bcast(&file->drho,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&file->nr,1,MPI_INT,0,world);
  MPI_Bcast(&file->dr,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&file->cut,1,MPI_DOUBLE,0,world);

  file->frho = (double *) memory->smalloc((file->nrho+1)*sizeof(double),
					  "pair:frho");
  file->rhor = (double *) memory->smalloc((file->nr+1)*sizeof(double),
					  "pair:rhor");
  file->zr = (double *) memory->smalloc((file->nr+1)*sizeof(double),
					"pair:zr");

  if (me == 0) grab(fp,file->nrho,&file->frho[1]);
  MPI_Bcast(&file->frho[1],file->nrho,MPI_DOUBLE,0,world);

  if (me == 0) grab(fp,file->nr,&file->zr[1]);
  MPI_Bcast(&file->zr[1],file->nr,MPI_DOUBLE,0,world);

  if (me == 0) grab(fp,file->nr,&file->rhor[1]);
  MPI_Bcast(&file->rhor[1],file->nr,MPI_DOUBLE,0,world);

  if (me == 0) fclose(fp);
}

/* ----------------------------------------------------------------------
   convert read-in funcfl potential(s) to standard array format
   interpolate all file values to a single grid and cutoff
------------------------------------------------------------------------- */

void PairEAM::file2array()
{
  int i,j,k,m,n;
  int ntypes = atom->ntypes;

  // determine max function params from all active funcfl files
  // active means some element is pointing at it via map

  int active;
  double rmax,rhomax;
  dr = drho = rmax = rhomax = 0.0;

  for (int i = 0; i < nfuncfl; i++) {
    active = 0;
    for (j = 1; j <= ntypes; j++)
      if (map[j] == i) active = 1;
    if (active == 0) continue;
    Funcfl *file = &funcfl[i];
    dr = MAX(dr,file->dr);
    drho = MAX(drho,file->drho);
    rmax = MAX(rmax,(file->nr-1) * file->dr);
    rhomax = MAX(rhomax,(file->nrho-1) * file->drho);
  }

  // set nr,nrho from cutoff and spacings
  // 0.5 is for round-off in divide

  nr = static_cast<int> (rmax/dr + 0.5);
  nrho = static_cast<int> (rhomax/drho + 0.5);

  // ------------------------------------------------------------------
  // setup frho arrays
  // ------------------------------------------------------------------

  // allocate frho arrays
  // nfrho = # of funcfl files + 1 for zero array
  
  nfrho = nfuncfl + 1;
  memory->destroy_2d_double_array(frho);
  frho = (double **) memory->create_2d_double_array(nfrho,nrho+1,"pair:frho");

  // interpolate each file's frho to a single grid and cutoff

  double r,p,cof1,cof2,cof3,cof4;
  
  n = 0;
  for (i = 0; i < nfuncfl; i++) {
    Funcfl *file = &funcfl[i];
    for (m = 1; m <= nrho; m++) {
      r = (m-1)*drho;
      p = r/file->drho + 1.0;
      k = static_cast<int> (p);
      k = MIN(k,file->nrho-2);
      k = MAX(k,2);
      p -= k;
      p = MIN(p,2.0);
      cof1 = -0.166666667*p*(p-1.0)*(p-2.0);
      cof2 = 0.5*(p*p-1.0)*(p-2.0);
      cof3 = -0.5*p*(p+1.0)*(p-2.0);
      cof4 = 0.166666667*p*(p*p-1.0);
      frho[n][m] = cof1*file->frho[k-1] + cof2*file->frho[k] + 
	cof3*file->frho[k+1] + cof4*file->frho[k+2];
    }
    n++;
  }

  // add extra frho of zeroes for non-EAM types to point to (pair hybrid)
  // this is necessary b/c fp is still computed for non-EAM atoms

  for (m = 1; m <= nrho; m++) frho[nfrho-1][m] = 0.0;

  // type2frho[i] = which frho array (0 to nfrho-1) each atom type maps to
  // if atom type doesn't point to file (non-EAM atom in pair hybrid)
  // then map it to last frho array of zeroes

  for (i = 1; i <= ntypes; i++)
    if (map[i] >= 0) type2frho[i] = map[i];
    else type2frho[i] = nfrho-1;

  // ------------------------------------------------------------------
  // setup rhor arrays
  // ------------------------------------------------------------------

  // allocate rhor arrays
  // nrhor = # of funcfl files

  nrhor = nfuncfl;
  memory->destroy_2d_double_array(rhor);
  rhor = (double **) memory->create_2d_double_array(nrhor,nr+1,"pair:rhor");

  // interpolate each file's rhor to a single grid and cutoff

  n = 0;
  for (i = 0; i < nfuncfl; i++) {
    Funcfl *file = &funcfl[i];
    for (m = 1; m <= nr; m++) {
      r = (m-1)*dr;
      p = r/file->dr + 1.0;
      k = static_cast<int> (p);
      k = MIN(k,file->nr-2);
      k = MAX(k,2);
      p -= k;
      p = MIN(p,2.0);
      cof1 = -0.166666667*p*(p-1.0)*(p-2.0);
      cof2 = 0.5*(p*p-1.0)*(p-2.0);
      cof3 = -0.5*p*(p+1.0)*(p-2.0);
      cof4 = 0.166666667*p*(p*p-1.0);
      rhor[n][m] = cof1*file->rhor[k-1] + cof2*file->rhor[k] +
	cof3*file->rhor[k+1] + cof4*file->rhor[k+2];
    }
    n++;
  }

  // type2rhor[i][j] = which rhor array (0 to nrhor-1) each type pair maps to
  // for funcfl files, I,J mapping only depends on I
  // OK if map = -1 (non-EAM atom in pair hybrid) b/c type2rhor not used

  for (i = 1; i <= ntypes; i++)
    for (j = 1; j <= ntypes; j++)
      type2rhor[i][j] = map[i];

  // ------------------------------------------------------------------
  // setup z2r arrays                                          // Question 8. what are z2r arrays? not clear?
  // ------------------------------------------------------------------

  // allocate z2r arrays
  // nz2r = N*(N+1)/2 where N = # of funcfl files

  nz2r = nfuncfl*(nfuncfl+1)/2;
  memory->destroy_2d_double_array(z2r);
  z2r = (double **) memory->create_2d_double_array(nz2r,nr+1,"pair:z2r");

  // create a z2r array for each file against other files, only for I >= J
  // interpolate zri and zrj to a single grid and cutoff

  double zri,zrj;

  n = 0;
  for (i = 0; i < nfuncfl; i++) {
    Funcfl *ifile = &funcfl[i];
    for (j = 0; j <= i; j++) {
      Funcfl *jfile = &funcfl[j];

      for (m = 1; m <= nr; m++) {
	r = (m-1)*dr;

	p = r/ifile->dr + 1.0;
	k = static_cast<int> (p);
	k = MIN(k,ifile->nr-2);
	k = MAX(k,2);
	p -= k;
	p = MIN(p,2.0);
	cof1 = -0.166666667*p*(p-1.0)*(p-2.0);
	cof2 = 0.5*(p*p-1.0)*(p-2.0);
	cof3 = -0.5*p*(p+1.0)*(p-2.0);
	cof4 = 0.166666667*p*(p*p-1.0);
	zri = cof1*ifile->zr[k-1] + cof2*ifile->zr[k] +
	  cof3*ifile->zr[k+1] + cof4*ifile->zr[k+2];

	p = r/jfile->dr + 1.0;
	k = static_cast<int> (p);
	k = MIN(k,jfile->nr-2);
	k = MAX(k,2);
	p -= k;
	p = MIN(p,2.0);
	cof1 = -0.166666667*p*(p-1.0)*(p-2.0);
	cof2 = 0.5*(p*p-1.0)*(p-2.0);
	cof3 = -0.5*p*(p+1.0)*(p-2.0);
	cof4 = 0.166666667*p*(p*p-1.0);
	zrj = cof1*jfile->zr[k-1] + cof2*jfile->zr[k] +
	  cof3*jfile->zr[k+1] + cof4*jfile->zr[k+2];

	z2r[n][m] = 27.2*0.529 * zri*zrj;
      }
      n++;
    }
  	// End loop  for setting up arrays
  }

  // type2z2r[i][j] = which z2r array (0 to nz2r-1) each type pair maps to
  // set of z2r arrays only fill lower triangular Nelement matrix
  // value = n = sum over rows of lower-triangular matrix until reach irow,icol
  // swap indices when irow < icol to stay lower triangular
  // OK if map = -1 (non-EAM atom in pair hybrid) b/c type2z2r not used

  int irow,icol;
  for (i = 1; i <= ntypes; i++) {
    for (j = 1; j <= ntypes; j++) {
      irow = map[i];
      icol = map[j];
      if (irow == -1 || icol == -1) continue;
      if (irow < icol) {
	irow = map[j];
	icol = map[i];
      }
      n = 0;
      for (m = 0; m < irow; m++) n += m + 1;
      n += icol;
      type2z2r[i][j] = n;
    }
  }
}

/* ---------------------------------------------------------------------- */

void PairEAM::array2spline()
{
  rdr = 1.0/dr;
  rdrho = 1.0/drho;

  memory->destroy_3d_double_array(frho_spline);
  memory->destroy_3d_double_array(rhor_spline);
  memory->destroy_3d_double_array(z2r_spline);
  memory->destroy_3d_double_array(u_ADP_spline);
  memory->destroy_3d_double_array(w_ADP_spline);

  frho_spline = memory->create_3d_double_array(nfrho,nrho+1,7,"pair:frho");
  rhor_spline = memory->create_3d_double_array(nrhor,nr+1,7,"pair:rhor");
  z2r_spline = memory->create_3d_double_array(nz2r,nr+1,7,"pair:z2r");
  u_ADP_spline = memory->create_3d_double_array(nu_ADP,nr+1,7,"pair:u_ADP");
  w_ADP_spline = memory->create_3d_double_array(nw_ADP,nr+1,7,"pair:w_ADP");

  for (int i = 0; i < nfrho; i++)
    interpolate(nrho,drho,frho[i],frho_spline[i]);

  for (int i = 0; i < nrhor; i++)
    interpolate(nr,dr,rhor[i],rhor_spline[i]);

  for (int i = 0; i < nz2r; i++)
    interpolate(nr,dr,z2r[i],z2r_spline[i]);
  
  for (int i = 0; i < nu_ADP; i++)
    interpolate(nr,dr,u_ADP[i],u_ADP_spline[i]);
  
  for (int i = 0; i < nw_ADP; i++)
    interpolate(nr,dr,w_ADP[i],w_ADP_spline[i]);
}

/* ---------------------------------------------------------------------- */

void PairEAM::interpolate(int n, double delta, double *f, double **spline)
{
  for (int m = 1; m <= n; m++) spline[m][6] = f[m];

  spline[1][5] = spline[2][6] - spline[1][6];
  spline[2][5] = 0.5 * (spline[3][6]-spline[1][6]);
  spline[n-1][5] = 0.5 * (spline[n][6]-spline[n-2][6]);
  spline[n][5] = spline[n][6] - spline[n-1][6];
  
  for (int m = 3; m <= n-2; m++)
    spline[m][5] = ((spline[m-2][6]-spline[m+2][6]) + 
		    8.0*(spline[m+1][6]-spline[m-1][6])) / 12.0;
  
  for (int m = 1; m <= n-1; m++) {
    spline[m][4] = 3.0*(spline[m+1][6]-spline[m][6]) - 
      2.0*spline[m][5] - spline[m+1][5];
    spline[m][3] = spline[m][5] + spline[m+1][5] - 
      2.0*(spline[m+1][6]-spline[m][6]);
  }
  
  spline[n][4] = 0.0;
  spline[n][3] = 0.0;
  
  for (int m = 1; m <= n; m++) {
    spline[m][2] = spline[m][5]/delta;
    spline[m][1] = 2.0*spline[m][4]/delta;
    spline[m][0] = 3.0*spline[m][3]/delta;
  }
}

/* ----------------------------------------------------------------------
   grab n values from file fp and put them in list
   values can be several to a line
   only called by proc 0
------------------------------------------------------------------------- */

void PairEAM::grab(FILE *fp, int n, double *list)
{
  char *ptr;
  char line[MAXLINE];

  int i = 0;
  while (i < n) {
    fgets(line,MAXLINE,fp);
    ptr = strtok(line," \t\n\r\f");
    list[i++] = atof(ptr);
    while (ptr = strtok(NULL," \t\n\r\f")) list[i++] = atof(ptr);
  }
}

/* ---------------------------------------------------------------------- */
//cout<<<"Into phi checking  "<<endl;
 // ofstream pair1_out("pair1_out.dat");

double PairEAM::single(int i, int j, int itype, int jtype,
		       double rsq, double factor_coul, double factor_lj,
		       double &fforce)
{
  int m;
  double r,p,rhoip,rhojp,z2,z2p,recip,phi,phip,psip;
  double *coeff;

 // 
// ofstream pair1_out("pair1_out.dat");
//

r = sqrt(rsq);
  p = r*rdr + 1.0;
  m = static_cast<int> (p);
  m = MIN(m,nr-1);
  p -= m;
  p = MIN(p,1.0);
  
  coeff = rhor_spline[type2rhor[itype][jtype]][m];
  rhoip = (coeff[0]*p + coeff[1])*p + coeff[2];
  coeff = rhor_spline[type2rhor[jtype][itype]][m];
  rhojp = (coeff[0]*p + coeff[1])*p + coeff[2];
  coeff = z2r_spline[type2z2r[itype][jtype]][m];
  z2p = (coeff[0]*p + coeff[1])*p + coeff[2];
  z2 = ((coeff[3]*p + coeff[4])*p + coeff[5])*p + coeff[6];

  recip = 1.0/r;
  phi = z2*recip;
  phip = z2p*recip - phi*recip;
  psip = fp[i]*rhojp + fp[j]*rhoip + phip;
  fforce = -psip*recip;

  return phi;   // Is it EAM energy?
//pair1_out<<i<<"  "<<phi<<endl;
//cout<<i<<"  "<<phi<<endl;
}

/* ---------------------------------------------------------------------- */

int PairEAM::pack_comm(int n, int *list, double *buf, int pbc_flag, int *pbc)
{
  int i,j,m;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    buf[m++] = fp[j];
  }
  return 1;
}

/* ---------------------------------------------------------------------- */

void PairEAM::unpack_comm(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) fp[i] = buf[m++];
}

/* ---------------------------------------------------------------------- */

int PairEAM::pack_reverse_comm(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) buf[m++] = rho[i];
  return 1;
}

/* ---------------------------------------------------------------------- */

void PairEAM::unpack_reverse_comm(int n, int *list, double *buf)
{
  int i,j,m;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    rho[j] += buf[m++];
  }
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays 
------------------------------------------------------------------------- */

double PairEAM::memory_usage()
{
  double bytes = maxeatom * sizeof(double);
  bytes += maxvatom*6 * sizeof(double);
  bytes += 2 * nmax * sizeof(double);
  return bytes;
}

/* ----------------------------------------------------------------------
   swap fp array with one passed in by caller
------------------------------------------------------------------------- */

void PairEAM::swap_eam(double *fp_caller, double **fp_caller_hold)
{
  double *tmp = fp;
  fp = fp_caller;
  *fp_caller_hold = tmp;
}


