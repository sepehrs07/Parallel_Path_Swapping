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

/***************************************************************
Implementation of Mishin's Angular Dependent Potential (ADP)
By: Chandra Veer Singh, Cornell University
----------------------------------------------------------------
The fitted functions rhor, z2r, frho, u_ADP, and w_ADP will be read from single element setfl file, and then 
interpolated for values at atoms, denoted by variables _spline, e.g., rhor_spline.
  * Original EAM input setfl file structure was:
	line 1:comment (ignored);
	line 2:atomic number, mass, lattice constant, lattice type (e.g. FCC);
    	line 3: Nrho, drho, Nr, dr, cutoff; 
	line 4 onwards: Tabulated values of following arrays
			1. embedding function F(rho) (Nrho values),
			2. density function rho(r) (Nr values),
			3. values of z2=r*phi 
  * New input setfl file structure is:
	line 1:comment (ignored);
	line 2:atomic number, mass, lattice constant, lattice type (e.g. FCC);
	line 3: Nrho, drho, Nr, dr, cutoff; 
	line 4 onwards: Tabulated values of following arrays
		        1. embedding function F(rho) (Nrho values),
			2. density function rho(r) (Nr values),
			3. values of z2=r*phi,		
			4. dipole potential function u_ADP (Nr values), 
			5. quadrupole potential function w_ADP (Nr values)		
Example: For binary alloy, the input setfl file should contain 13 functions and other info in following order: 
header,Elem1_info,Frho_1,rho_1,Elem2_info,Frho_2,rho_2,rphi_1,rphi_12,rphi_2 ==> EAM part (7 functions), then it must also contain
u_1,u_12,u_2,w_1,w_12,w_2 ==> ADP part (6 functions)
***************************************************************/

#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "pair_eam_alloy.h"
#include "atom.h"
#include "comm.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;

#define MAXLINE 1024

/* ---------------------------------------------------------------------- */

PairEAMAlloy::PairEAMAlloy(LAMMPS *lmp) : PairEAM(lmp)
{
  one_coeff = 1;
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
   read DYNAMO setfl file
------------------------------------------------------------------------- */

void PairEAMAlloy::coeff(int narg, char **arg)
{
  int i,j;

  if (!allocated) allocate();

  if (narg != 3 + atom->ntypes)
    error->all("Incorrect args for pair coefficients");

  // insure I,J args are * *

  if (strcmp(arg[0],"*") != 0 || strcmp(arg[1],"*") != 0)
    error->all("Incorrect args for pair coefficients");

  // read EAM setfl file

  if (setfl) {
    for (i = 0; i < setfl->nelements; i++) delete [] setfl->elements[i];
    delete [] setfl->elements;
    delete [] setfl->mass;
    memory->destroy_2d_double_array(setfl->frho);
    memory->destroy_2d_double_array(setfl->rhor);
    memory->destroy_3d_double_array(setfl->z2r);
    memory->destroy_3d_double_array(setfl->u_ADP); // u_ADP
    memory->destroy_3d_double_array(setfl->w_ADP);  // w_ADP
    delete setfl;
  }
  setfl = new Setfl();
  read_file(arg[2]);

  // read args that map atom types to elements in potential file
  // map[i] = which element the Ith atom type is, -1 if NULL

  for (i = 3; i < narg; i++) {
    if (strcmp(arg[i],"NULL") == 0) {
      map[i-2] = -1;
      continue;
    }
    for (j = 0; j < setfl->nelements; j++)
      if (strcmp(arg[i],setfl->elements[j]) == 0) break;
    if (j < setfl->nelements) map[i-2] = j;
    else error->all("No matching element in EAM potential file");
  }

  // clear setflag since coeff() called once with I,J = * *

  int n = atom->ntypes;
  for (i = 1; i <= n; i++)
    for (j = i; j <= n; j++)
      setflag[i][j] = 0;

  // set setflag i,j for type pairs where both are mapped to elements
  // set mass of atom type if i = j

  int count = 0;
  for (i = 1; i <= n; i++) {
    for (j = i; j <= n; j++) {
      if (map[i] >= 0 && map[j] >= 0) {
	setflag[i][j] = 1;
	if (i == j) atom->set_mass(i,setfl->mass[map[i]]);
	count++;
      }
    }
  }

  if (count == 0) error->all("Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
   read a multi-element DYNAMO setfl file
------------------------------------------------------------------------- */

void PairEAMAlloy::read_file(char *filename)
{
  Setfl *file = setfl;

  // open potential file

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
  // read and broadcast header
  // extract element names from nelements line
  int n;
  if (me == 0) {
    fgets(line,MAXLINE,fp);
    fgets(line,MAXLINE,fp);
    fgets(line,MAXLINE,fp);
    fgets(line,MAXLINE,fp);
    n = strlen(line) + 1;
  }
  MPI_Bcast(&n,1,MPI_INT,0,world);
  MPI_Bcast(line,n,MPI_CHAR,0,world);

  sscanf(line,"%d",&file->nelements);
//	pot_check<<"Number of atom types="<<file->nelements<<"\n";
  int nwords = atom->count_words(line);
  if (nwords != file->nelements + 1)
    error->all("Incorrect element names in EAM potential file");
  
  char **words = new char*[file->nelements+1];
  nwords = 0;
  char *first = strtok(line," \t\n\r\f");
  while (words[nwords++] = strtok(NULL," \t\n\r\f")) continue;

  file->elements = new char*[file->nelements];
  for (int i = 0; i < file->nelements; i++) {
    n = strlen(words[i]) + 1;
    file->elements[i] = new char[n];
    strcpy(file->elements[i],words[i]);
  }
  delete [] words;

  if (me == 0) {
    fgets(line,MAXLINE,fp);
    sscanf(line,"%d %lg %d %lg %lg",
	   &file->nrho,&file->drho,&file->nr,&file->dr,&file->cut);
  }
  
  MPI_Bcast(&file->nrho,1,MPI_INT,0,world);
  MPI_Bcast(&file->drho,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&file->nr,1,MPI_INT,0,world);
  MPI_Bcast(&file->dr,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&file->cut,1,MPI_DOUBLE,0,world);

  file->mass = new double[file->nelements];
  file->frho = memory->create_2d_double_array(file->nelements,file->nrho+1,
					      "pair:frho");
  file->rhor = memory->create_2d_double_array(file->nelements,file->nr+1,
					      "pair:rhor");
  file->z2r = memory->create_3d_double_array(file->nelements,file->nelements,
					     file->nr+1,"pair:z2r");
  file->u_ADP = memory->create_3d_double_array(file->nelements,file->nelements,	
					     file->nr+1, "pair:u_ADP");			// u_ADP
  file->w_ADP = memory->create_3d_double_array(file->nelements,file->nelements,	
					     file->nr+1, "pair:w_ADP");			// w_ADP
  int i,j,tmp;
  for (i = 0; i < file->nelements; i++) {
    if (me == 0) {
      fgets(line,MAXLINE,fp);
      sscanf(line,"%d %lg",&tmp,&file->mass[i]);
    }
    MPI_Bcast(&file->mass[i],1,MPI_DOUBLE,0,world);

    if (me == 0) grab(fp,file->nrho,&file->frho[i][1]);
    MPI_Bcast(&file->frho[i][1],file->nrho,MPI_DOUBLE,0,world);
    if (me == 0) grab(fp,file->nr,&file->rhor[i][1]);
    MPI_Bcast(&file->rhor[i][1],file->nr,MPI_DOUBLE,0,world);
}

  for (i = 0; i < file->nelements; i++)
    for (j = 0; j <= i; j++) {
      if (me == 0) grab(fp,file->nr,&file->z2r[i][j][1]);
      MPI_Bcast(&file->z2r[i][j][1],file->nr,MPI_DOUBLE,0,world);
    }
 // For u_ADP 
  for (i = 0; i < file->nelements; i++) {
    for (j = 0; j <= i; j++) {
      if (me == 0) grab(fp,file->nr,&file->u_ADP[i][j][1]);
      MPI_Bcast(&file->u_ADP[i][j][1],file->nr,MPI_DOUBLE,0,world);			// u_ADP
    }
}
 // For w_ADP 
  for (i = 0; i < file->nelements; i++) {
    for (j = 0; j <= i; j++) {
      if (me == 0) grab(fp,file->nr,&file->w_ADP[i][j][1]);
      MPI_Bcast(&file->w_ADP[i][j][1],file->nr,MPI_DOUBLE,0,world);			// w_ADP
    }
}
 // close the potential file
  if (me == 0) fclose(fp);
}

/* ----------------------------------------------------------------------
   copy read-in setfl potential to standard array format
------------------------------------------------------------------------- */

void PairEAMAlloy::file2array()
{
  int i,j,m,n;
  int ntypes = atom->ntypes;

  // set function params directly from setfl file

  nrho = setfl->nrho;
  nr = setfl->nr;
  drho = setfl->drho;
  dr = setfl->dr;

  // ------------------------------------------------------------------
  // setup frho arrays
  // ------------------------------------------------------------------

  // allocate frho arrays
  // nfrho = # of setfl elements + 1 for zero array
  
  nfrho = setfl->nelements + 1;
  memory->destroy_2d_double_array(frho);
  frho = (double **) memory->create_2d_double_array(nfrho,nrho+1,"pair:frho");

  // copy each element's frho to global frho

  for (i = 0; i < setfl->nelements; i++)
    for (m = 1; m <= nrho; m++) {
	frho[i][m] = setfl->frho[i][m];

}
// add extra frho of zeroes for non-EAM types to point to (pair hybrid)
  // this is necessary b/c fp is still computed for non-EAM atoms

  for (m = 1; m <= nrho; m++) frho[nfrho-1][m] = 0.0;

  // type2frho[i] = which frho array (0 to nfrho-1) each atom type maps to
  // if atom type doesn't point to element (non-EAM atom in pair hybrid)
  // then map it to last frho array of zeroes

  for (i = 1; i <= ntypes; i++)
    if (map[i] >= 0) type2frho[i] = map[i];
    else type2frho[i] = nfrho-1;

  // ------------------------------------------------------------------
  // setup rhor arrays
  // ------------------------------------------------------------------

  // allocate rhor arrays
  // nrhor = # of setfl elements

  nrhor = setfl->nelements;
  memory->destroy_2d_double_array(rhor);
  rhor = (double **) memory->create_2d_double_array(nrhor,nr+1,"pair:rhor");

  // copy each element's rhor to global rhor

  for (i = 0; i < setfl->nelements; i++)
    for (m = 1; m <= nr; m++) {
	rhor[i][m] = setfl->rhor[i][m];

}
// type2rhor[i][j] = which rhor array (0 to nrhor-1) each type pair maps to
  // for setfl files, I,J mapping only depends on I
  // OK if map = -1 (non-EAM atom in pair hybrid) b/c type2rhor not used

  for (i = 1; i <= ntypes; i++)
    for (j = 1; j <= ntypes; j++)
      type2rhor[i][j] = map[i];

  // ------------------------------------------------------------------
  // setup z2r arrays
  // ------------------------------------------------------------------

  // allocate z2r arrays
  // nz2r = N*(N+1)/2 where N = # of setfl elements

  nz2r = setfl->nelements * (setfl->nelements+1) / 2;
  memory->destroy_2d_double_array(z2r);
  z2r = (double **) memory->create_2d_double_array(nz2r,nr+1,"pair:z2r");

  // copy each element pair z2r to global z2r, only for I >= J

  n = 0;
  for (i = 0; i < setfl->nelements; i++)
    for (j = 0; j <= i; j++) {
      for (m = 1; m <= nr; m++) {
	z2r[n][m] = setfl->z2r[i][j][m];
	}
	n++;
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

  // ------------------------------------------------------------------
  // setup u_ADP arrays									\\ u_ADP
  // ------------------------------------------------------------------

  // allocate u_ADP arrays
  // nu_ADP = N*(N+1)/2 where N= # of setfl elements

  nu_ADP = setfl->nelements * (setfl->nelements+1) / 2;
  memory->destroy_2d_double_array(u_ADP);
  u_ADP = (double **) memory->create_2d_double_array(nu_ADP,nr+1,"pair:u_ADP");

  // copy each element's u_ADP to global u_ADP
  n=0;
  for (i = 0; i < setfl->nelements; i++)
    for (j = 0; j <= i; j++) {
      for (m = 1; m <= nr; m++) {
	u_ADP[n][m] = setfl->u_ADP[i][j][m];
	}
	n++;
    } 
  
  // type2u_ADP[i][j] = which u_ADP array (0 to nu_ADP-1) each type pair maps to
  // set of u_ADP arrays only fill lower triangular Nelement matrix
  // value = n = sum over rows of lower-triangular matrix until reach irow,icol
  // swap indices when irow < icol to stay lower triangular
  // OK if map = -1 (non-EAM atom in pair hybrid) b/c type2u_ADP not used

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
      type2u_ADP[i][j] = n;
    }
  }

  // ------------------------------------------------------------------
  // setup w_ADP arrays									\\ w_ADP
  // ------------------------------------------------------------------

  // allocate w_ADP arrays
  // nw_ADP = N*(N+1)/2 where N= # of setfl elements

  nw_ADP = setfl->nelements * (setfl->nelements+1) / 2;
  memory->destroy_2d_double_array(w_ADP);
  w_ADP = (double **) memory->create_2d_double_array(nw_ADP,nr+1,"pair:w_ADP");

  // copy each element's w_ADP to global w_ADP
  n=0;
  for (i = 0; i < setfl->nelements; i++)
    for (j = 0; j <= i; j++) {
      for (m = 1; m <= nr; m++) {
	w_ADP[n][m] = setfl->w_ADP[i][j][m];
	}
	n++;
    } 
  
  // type2w_ADP[i][j] = which w_ADP array (0 to nw_ADP-1) each type pair maps to
  // set of w_ADP arrays only fill lower triangular Nelement matrix
  // value = n = sum over rows of lower-triangular matrix until reach irow,icol
  // swap indices when irow < icol to stay lower triangular
  // OK if map = -1 (non-EAM atom in pair hybrid) b/c type2w_ADP not used

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
      type2w_ADP[i][j] = n;
    }
  }
}
