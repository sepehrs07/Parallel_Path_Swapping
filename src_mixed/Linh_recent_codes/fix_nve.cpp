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
#include "stdio.h"
#include "string.h"
#include "fix_nve.h"
#include "atom.h"
#include "force.h"
#include "update.h"
#include "respa.h"
#include "error.h"
#include "iostream.h"
#include "fstream.h"
#include "math.h"
#include "random_mars.h"
#include "iomanip.h"
#include "universe.h"

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

using namespace LAMMPS_NS;


/* ---------------------------------------------------------------------- */

FixNVE::FixNVE(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (narg != 3) error->all("Illegal fix nve command");
}

/* ---------------------------------------------------------------------- */

int FixNVE::setmask()
{
  int mask = 0;
  mask |= INITIAL_INTEGRATE;
  mask |= FINAL_INTEGRATE;
  mask |= INITIAL_INTEGRATE_RESPA;
  mask |= FINAL_INTEGRATE_RESPA;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixNVE::init()
{
  dtv = update->dt;
  dtf = update->dt * force->ftm2v;

  if (strcmp(update->integrate_style,"respa") == 0)
    step_respa = ((Respa *) update->integrate)->step;
}


 /* FTS variables */
 static int    pmf_call_counter=0;
 static int         ave_counter=0;
 static int    constraint;
 static double cosine;
 static double xlo, xhi, ylo, yhi, zlo, zhi;
 static double BL[3], orig_com[3], com[3];

 static double     add_force[500000];
 static double     atom_mass[500000];
 static double     sample_mass;

 static int            Vid[500000];
 static double       str_x[500000*3];
 static double   old_str_x[500000*3];
 static double trial_str_x[500000*3];

 static double   pmf_vec[500000*3];

 static double     old_x[500000][3];
 static double      uw_x[500000][3];
 static double   trial_x[500000][3];
 static double   trial_v[500000][3];
 static double     ave_x[500000][3];
 static double    orig_x[500000][3];
 static double  anchor_x[500000][3];
 static double      g_id[500000];

 static double ave_pforce;
 static double pmf_dist;
 static double pmf_force_scaler=0, last_scaler;
 static double pmf_drift_correct,pmf_vec_org;
 static double orig_string_length=1.0e+10;
 static double pmf_vec_length, pmf_vec_length_TOTAL, pmf_v_l_total;
 double rand_U1, rand_U2, rand_G;

 static double delta_tao=0.1;
 static double delta_t=0.01;
 
 static int NN=60;
 static int me_flux[60];             //-----NN
 static int nb_cnt=6;

 static int    total_run=10000000;
 static int     MD_start=10000000;
 static int       cvg_id=10000000;
 static int     dump_img=100000;
 static int  dump_matrix=100000;
 static int   dump_force=100000;
 static int dump_spacing=10000;


/* ----------------------------------------------------------------------
   allow for both per-type and per-atom mass
------------------------------------------------------------------------- */

void FixNVE::initial_integrate(int vflag)
{

  /* PMF variables */

  double vor_dist[2*nb_cnt+1];

  static int damping;
  double temp, temp1, trash, x_scaler=0;
  double accel;
  int pcheck, reparam_flag; 
  double distance, project_length, vor_check;
  int i, j, k, m, q, lj_marker, rand_index;
  double lnhuge=1e+30;
  double lj, kn, bb;
  double img_spacing, string_length, min_length;
  double pforce_dot_prod, temp_pforce, temp_step;

  double *nb0_str_x;
  double *nb5_str_x;
  double *nb6_str_x;
  double *nb_lj1_x;
  double *nb_lj0_x;

  double *nb1_l_x;
  double *nb2_l_x;
  double *nb3_l_x;
  double *nb4_l_x;
  double *nb5_l_x;
  double *nb6_l_x;
  double *nb1_r_x;
  double *nb2_r_x;
  double *nb3_r_x;
  double *nb4_r_x;
  double *nb5_r_x;
  double *nb6_r_x;

  double length_along_line[NN], g1_spacing[NN], g2_spacing[NN], g_ave_pforce[NN];    //-NN
  int g_marker[NN];                                                                             //-NN

  /*-----------------*/

  double dtfm;
  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double *mass = atom->mass;
  double *rmass = atom->rmass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  MPI_Status status;
  int me;
  MPI_Comm_rank(universe->uworld, &me);

 
  /*------------- First step:  read box sizes, set constraint on/off, and unwrap periodic coords to match inital state (0) ------------*/
    if(pmf_call_counter==0){
      /* read if constraint is on/off and box sizes */
      ifstream vector("pos.read");
      vector>>xlo>>xhi;
      vector>>ylo>>yhi;
      vector>>zlo>>zhi;
      vector>>sample_mass;
      for(i=0;i<nlocal;i++)  vector>>trash>>atom_mass[i]>>add_force[i];      // for adding force
      vector.close();
      BL[0] = xhi-xlo;
      BL[1] = yhi-ylo;
      BL[2] = zhi-zlo;

      /* read Voronoid cells definition */
      ifstream vcell("vcell.def");
      for(i=0;i<nlocal;i++)  vcell>>trash>>Vid[i];
      vcell.close();
 
      /* assign images to nodes */
      char buf[128];
      sprintf(buf,"R%d.data",me);
      ifstream indata (buf);
      indata>>trash>>orig_com[0]>>orig_com[1]>>orig_com[2];
      for(i=0;i<nlocal;i++)   indata>>g_id[i]>>uw_x[i][0]>>uw_x[i][1]>>uw_x[i][2];
      indata.close();

      for(i=0;i<nlocal;i++) for(k=0;k<3;k++) { old_x[i][k] = orig_x[i][k] = x[i][k]; }
      for(i=0;i<nlocal;i++) for(k=0;k<3;k++) { v[i][k] = 0.0; }

      /* initiate rejection counter */
      for(m=0;m<NN;m++) me_flux[m]=0;
    }


  /*----------------------------------------------------- For all steps ------------------------------------------------------*/

  if(pmf_call_counter>0){

    if(pmf_call_counter==1) {
      // periodicity in x
      for (i=0;i<nlocal;i++) {
        temp=x[i][0]-orig_x[i][0];
        if (temp>BL[0]*0.8)  { temp -= BL[0];
        } else if(-temp>BL[0]*0.8) { temp += BL[0]; }
        x[i][0] = temp + orig_x[i][0];
      } 
      // periodicity in z
      for (i=0;i<nlocal;i++) {
        temp=x[i][2]-orig_x[i][2];
        if (temp>BL[2]*0.8)  { temp -= BL[2];
        } else if(-temp>BL[2]*0.8) { temp += BL[2]; }
        x[i][2] = temp + orig_x[i][2];
      }
      for(i=0;i<nlocal;i++) for(k=0;k<3;k++) orig_x[i][k] = x[i][k];
      for(k=0;k<3;k++) {
        orig_com[k]=0.0;
        for(i=0;i<nlocal;i++)    orig_com[k] += orig_x[i][k]*atom_mass[i];
        orig_com[k] /= sample_mass;
      }
    }

    // adding shear force
//    for (int i = 0; i < nlocal; i++)    f[i][0] += add_force[i];

    /* change from updated x to unwrapped coordinates (to match inital state) */
    // periodicity in x
    for (i=0;i<nlocal;i++) {
      temp=x[i][0]-orig_x[i][0];
      if (temp>BL[0]*0.8)  { temp -= BL[0];
      } else if(-temp>BL[0]*0.8) { temp += BL[0]; }
      uw_x[i][0] = temp + orig_x[i][0];
    }
    // periodicity in z
    for (i=0;i<nlocal;i++) {
      temp=x[i][2]-orig_x[i][2];
      if (temp>BL[2]*0.8)  { temp -= BL[2];
      } else if(-temp>BL[2]*0.8) { temp += BL[2]; }
      uw_x[i][2] = temp + orig_x[i][2];
    }
    // free in y
    for (i=0;i<nlocal;i++)   uw_x[i][1] = x[i][1];

    if(pmf_call_counter==1) { for(i=0;i<nlocal;i++)   for(k=0;k<3;k++)  str_x[3*i+k]=ave_x[i][k]=uw_x[i][k]; }
    for(i=0;i<nlocal;i++)   for(k=0;k<3;k++)   old_x[i][k] = uw_x[i][k];


    if(pmf_call_counter==1) {
      nb0_str_x = new double [3*nlocal];
      MPI_Barrier (universe->uworld);
      if (me==0) {
        MPI_Send(str_x, 3*nlocal, MPI_DOUBLE, me+1, me+1, universe->uworld);
        img_spacing=0.0;
      }else{
        if (me==NN-1) MPI_Recv(nb0_str_x, 3*nlocal, MPI_DOUBLE, me-1, me, universe->uworld, &status);
        else MPI_Sendrecv(str_x, 3*nlocal, MPI_DOUBLE, me+1, me+1, nb0_str_x, 3*nlocal, MPI_DOUBLE, me-1, me, universe->uworld, &status);    // send to the right, receive from the left
        img_spacing=0.0;
        for(i=0;i<nlocal;i++) if(Vid[i]==1) {
          k=0;
          temp = str_x[3*i+k] - nb0_str_x[3*i+k];
          img_spacing += temp*temp;
          k=1;
          temp = str_x[3*i+k] - nb0_str_x[3*i+k];
          img_spacing += temp*temp;
          k=2;
          temp = str_x[3*i+k] - nb0_str_x[3*i+k];
          img_spacing += temp*temp;
        }
        img_spacing = sqrt(img_spacing);
      }
      MPI_Barrier (universe->uworld);
      MPI_Allgather(&img_spacing, 1, MPI_DOUBLE, g1_spacing, 1, MPI_DOUBLE, universe->uworld);
      length_along_line[0]=0.0;
      for(j=1;j<NN;j++)   length_along_line[j] = g1_spacing[j] + length_along_line[j-1];
      orig_string_length=length_along_line[NN-1];
      delete [] nb0_str_x;
    }


    /* Step 1: Propose new uw_x and check with Voronoi cell to update wu_x */
    // Numerical sampling
    for(j=0;j<me+1;j++)  { rand_U1 = (double)rand(); rand_U2 = (double)rand(); }
      for(i=0;i<nlocal;i++) {
        for(k=0;k<3;k++) {
          rand_U1 = (double)rand()/(double)RAND_MAX;
          rand_U2 = (double)rand()/(double)RAND_MAX;
          if (rand_U1>1e-30)   rand_G=sqrt(-2*log(rand_U1))*cos(2*3.14159*rand_U2);
          else  rand_G=0.0;
          trial_x[i][k] = uw_x[i][k] + delta_t*f[i][k] + sqrt(2*0.02585*delta_t)*rand_G;
        }
      }
#if 0
      // MD sampling
      if (rmass) { 
        for (i=0; i<nlocal; i++) {
          if (mask[i] & groupbit) {
            dtfm = dtf / rmass[i];
            trial_v[i][0] = v[i][0] + dtfm*f[i][0];
            trial_v[i][1] = v[i][1] + dtfm*f[i][1];
            trial_v[i][2] = v[i][2] + dtfm*f[i][2];
            trial_x[i][0] = uw_x[i][0] + dtv*trial_v[i][0];
            trial_x[i][1] = uw_x[i][1] + dtv*trial_v[i][1];
            trial_x[i][2] = uw_x[i][2] + dtv*trial_v[i][2];
          }
        }
      } else {
        for (i=0; i<nlocal; i++) {
          if (mask[i] & groupbit) {
            dtfm = dtf / mass[type[i]];
            trial_v[i][0] = v[i][0] + dtfm*f[i][0];
            trial_v[i][1] = v[i][1] + dtfm*f[i][1];
            trial_v[i][2] = v[i][2] + dtfm*f[i][2];
            trial_x[i][0] = uw_x[i][0] + dtv*trial_v[i][0];
            trial_x[i][1] = uw_x[i][1] + dtv*trial_v[i][1];
            trial_x[i][2] = uw_x[i][2] + dtv*trial_v[i][2];
          }
        }
      } 
#endif

    /* COM correction */
    for(k=0;k<3;k++) {
      com[k]=0.0;
      for(i=0;i<nlocal;i++)    com[k] += trial_x[i][k]*atom_mass[i];
      com[k] /= sample_mass;
    } 
    for(i=0;i<nlocal;i++)   for(k=0;k<3;k++)   trial_x[i][k]-=(com[k]-orig_com[k]);

    /* Swapping string coords before convergence */
      // send to +6; receive from -6
      nb6_l_x = new double [3*nlocal];
      MPI_Barrier (universe->uworld);
      if (me<6) {
        MPI_Send(str_x, 3*nlocal, MPI_DOUBLE, me+6, me+6, universe->uworld);
      }else{
        if (me>NN-(6+1)) MPI_Recv(nb6_l_x, 3*nlocal, MPI_DOUBLE, me-6, me, universe->uworld, &status);
        else MPI_Sendrecv(str_x, 3*nlocal, MPI_DOUBLE, me+6, me+6, nb6_l_x, 3*nlocal, MPI_DOUBLE, me-6, me, universe->uworld, &status);
      }
      // send to -6; receive from +6
      nb6_r_x = new double [3*nlocal];
      MPI_Barrier (universe->uworld);
      if (me>NN-(6+1)) {
        MPI_Send(str_x, 3*nlocal, MPI_DOUBLE, me-6, me-6, universe->uworld);
      }else{
        if (me<6) MPI_Recv(nb6_r_x, 3*nlocal, MPI_DOUBLE, me+6, me, universe->uworld, &status);
        else MPI_Sendrecv(str_x, 3*nlocal, MPI_DOUBLE, me-6, me-6, nb6_r_x, 3*nlocal, MPI_DOUBLE, me+6, me, universe->uworld, &status);
      }

      // send to +5; receive from -5
      nb5_l_x = new double [3*nlocal];
      MPI_Barrier (universe->uworld);
      if (me<5) {
        MPI_Send(str_x, 3*nlocal, MPI_DOUBLE, me+5, me+5, universe->uworld);
      }else{
        if (me>NN-(5+1)) MPI_Recv(nb5_l_x, 3*nlocal, MPI_DOUBLE, me-5, me, universe->uworld, &status);
        else MPI_Sendrecv(str_x, 3*nlocal, MPI_DOUBLE, me+5, me+5, nb5_l_x, 3*nlocal, MPI_DOUBLE, me-5, me, universe->uworld, &status);
      }
      // send to -5; receive from +5
      nb5_r_x = new double [3*nlocal];
      MPI_Barrier (universe->uworld);
      if (me>NN-(5+1)) {
        MPI_Send(str_x, 3*nlocal, MPI_DOUBLE, me-5, me-5, universe->uworld);
      }else{
        if (me<5) MPI_Recv(nb5_r_x, 3*nlocal, MPI_DOUBLE, me+5, me, universe->uworld, &status);
        else MPI_Sendrecv(str_x, 3*nlocal, MPI_DOUBLE, me-5, me-5, nb5_r_x, 3*nlocal, MPI_DOUBLE, me+5, me, universe->uworld, &status);
      }

      // send to +4; receive from -4
      nb4_l_x = new double [3*nlocal];
      MPI_Barrier (universe->uworld);
      if (me<4) {
        MPI_Send(str_x, 3*nlocal, MPI_DOUBLE, me+4, me+4, universe->uworld);
      }else{
        if (me>NN-(4+1)) MPI_Recv(nb4_l_x, 3*nlocal, MPI_DOUBLE, me-4, me, universe->uworld, &status);
        else MPI_Sendrecv(str_x, 3*nlocal, MPI_DOUBLE, me+4, me+4, nb4_l_x, 3*nlocal, MPI_DOUBLE, me-4, me, universe->uworld, &status);
      }
      // send to -4; receive from +4
      nb4_r_x = new double [3*nlocal];
      MPI_Barrier (universe->uworld);
      if (me>NN-(4+1)) {
        MPI_Send(str_x, 3*nlocal, MPI_DOUBLE, me-4, me-4, universe->uworld);
      }else{
        if (me<4) MPI_Recv(nb4_r_x, 3*nlocal, MPI_DOUBLE, me+4, me, universe->uworld, &status);
        else MPI_Sendrecv(str_x, 3*nlocal, MPI_DOUBLE, me-4, me-4, nb4_r_x, 3*nlocal, MPI_DOUBLE, me+4, me, universe->uworld, &status);
      }

      // send to +3; receive from -3
      nb3_l_x = new double [3*nlocal];
      MPI_Barrier (universe->uworld);
      if (me<3) {
        MPI_Send(str_x, 3*nlocal, MPI_DOUBLE, me+3, me+3, universe->uworld);
      }else{
        if (me>NN-(3+1)) MPI_Recv(nb3_l_x, 3*nlocal, MPI_DOUBLE, me-3, me, universe->uworld, &status);
        else MPI_Sendrecv(str_x, 3*nlocal, MPI_DOUBLE, me+3, me+3, nb3_l_x, 3*nlocal, MPI_DOUBLE, me-3, me, universe->uworld, &status);
      }
      // send to -3; receive from +3
      nb3_r_x = new double [3*nlocal];
      MPI_Barrier (universe->uworld);
      if (me>NN-(3+1)) {
        MPI_Send(str_x, 3*nlocal, MPI_DOUBLE, me-3, me-3, universe->uworld);
      }else{
        if (me<3) MPI_Recv(nb3_r_x, 3*nlocal, MPI_DOUBLE, me+3, me, universe->uworld, &status);
        else MPI_Sendrecv(str_x, 3*nlocal, MPI_DOUBLE, me-3, me-3, nb3_r_x, 3*nlocal, MPI_DOUBLE, me+3, me, universe->uworld, &status);
      }

      // send to +2; receive from -2
      nb2_l_x = new double [3*nlocal];
      MPI_Barrier (universe->uworld);
      if (me<2) {
        MPI_Send(str_x, 3*nlocal, MPI_DOUBLE, me+2, me+2, universe->uworld);
      }else{
        if (me>NN-(2+1)) MPI_Recv(nb2_l_x, 3*nlocal, MPI_DOUBLE, me-2, me, universe->uworld, &status);
        else MPI_Sendrecv(str_x, 3*nlocal, MPI_DOUBLE, me+2, me+2, nb2_l_x, 3*nlocal, MPI_DOUBLE, me-2, me, universe->uworld, &status);
      }
      // send to -2; receive from +2
      nb2_r_x = new double [3*nlocal];
      MPI_Barrier (universe->uworld);
      if (me>NN-(2+1)) {
        MPI_Send(str_x, 3*nlocal, MPI_DOUBLE, me-2, me-2, universe->uworld);
      }else{
        if (me<2) MPI_Recv(nb2_r_x, 3*nlocal, MPI_DOUBLE, me+2, me, universe->uworld, &status);
        else MPI_Sendrecv(str_x, 3*nlocal, MPI_DOUBLE, me-2, me-2, nb2_r_x, 3*nlocal, MPI_DOUBLE, me+2, me, universe->uworld, &status);
      }

      // send to +1; receive from -1
      nb1_l_x = new double [3*nlocal];
      MPI_Barrier (universe->uworld);
      if (me<1) {
        MPI_Send(str_x, 3*nlocal, MPI_DOUBLE, me+1, me+1, universe->uworld);
      }else{
        if (me>NN-(1+1)) MPI_Recv(nb1_l_x, 3*nlocal, MPI_DOUBLE, me-1, me, universe->uworld, &status);
        else MPI_Sendrecv(str_x, 3*nlocal, MPI_DOUBLE, me+1, me+1, nb1_l_x, 3*nlocal, MPI_DOUBLE, me-1, me, universe->uworld, &status);
      }
      // send to -1; receive from +1
      nb1_r_x = new double [3*nlocal];
      MPI_Barrier (universe->uworld);
      if (me>NN-(1+1)) {
        MPI_Send(str_x, 3*nlocal, MPI_DOUBLE, me-1, me-1, universe->uworld);
      }else{
        if (me<1) MPI_Recv(nb1_r_x, 3*nlocal, MPI_DOUBLE, me+1, me, universe->uworld, &status);
        else MPI_Sendrecv(str_x, 3*nlocal, MPI_DOUBLE, me-1, me-1, nb1_r_x, 3*nlocal, MPI_DOUBLE, me+1, me, universe->uworld, &status);
      }



    /* distance to the center Voronoi cell */
    vor_dist[0+nb_cnt]=0.0;
    for(i=0;i<nlocal;i++)  if(Vid[i]==1) {
      k=0;
      temp = trial_x[i][k] - str_x[3*i+k];
      vor_dist[0+nb_cnt] += temp*temp;
      k=1;
      temp = trial_x[i][k] - str_x[3*i+k];
      vor_dist[0+nb_cnt] += temp*temp;
      k=2; 
      temp = trial_x[i][k] - str_x[3*i+k]; 
      vor_dist[0+nb_cnt] += temp*temp;
    }
    vor_dist[0+nb_cnt]=sqrt(vor_dist[0+nb_cnt]);


    /* calculating distances to Voronoid cells */

      // distance from neighbor -6
      if (me<6) { vor_dist[-6+nb_cnt] = lnhuge;
      }else{
        vor_dist[-6+nb_cnt]=0.0;
        for(i=0;i<nlocal;i++)  if(Vid[i]==1) {
          k=0;
          temp = trial_x[i][k] - nb6_l_x[3*i+k];
          vor_dist[-6+nb_cnt] += temp*temp;
          k=1;
          temp = trial_x[i][k] - nb6_l_x[3*i+k];
          vor_dist[-6+nb_cnt] += temp*temp;
          k=2;
          temp = trial_x[i][k] - nb6_l_x[3*i+k];
          vor_dist[-6+nb_cnt] += temp*temp;
        }
        vor_dist[-6+nb_cnt]=sqrt(vor_dist[-6+nb_cnt]);
      }
      // distance from neighbor +6
      if (me>NN-(6+1)) { vor_dist[6+nb_cnt] = lnhuge;
      }else{
        vor_dist[6+nb_cnt]=0.0;
        for(i=0;i<nlocal;i++)  if(Vid[i]==1) {
          k=0;
          temp = trial_x[i][k] - nb6_r_x[3*i+k];
          vor_dist[6+nb_cnt] += temp*temp;
          k=1;
          temp = trial_x[i][k] - nb6_r_x[3*i+k];
          vor_dist[6+nb_cnt] += temp*temp;
          k=2;
          temp = trial_x[i][k] - nb6_r_x[3*i+k];
          vor_dist[6+nb_cnt] += temp*temp;
        }
        vor_dist[6+nb_cnt]=sqrt(vor_dist[6+nb_cnt]);
      }

      // distance from neighbor -5
      if (me<5) { vor_dist[-5+nb_cnt] = lnhuge;
      }else{
        vor_dist[-5+nb_cnt]=0.0;
        for(i=0;i<nlocal;i++)  if(Vid[i]==1) {
          k=0;
          temp = trial_x[i][k] - nb5_l_x[3*i+k];
          vor_dist[-5+nb_cnt] += temp*temp;
          k=1;
          temp = trial_x[i][k] - nb5_l_x[3*i+k];
          vor_dist[-5+nb_cnt] += temp*temp;
          k=2;
          temp = trial_x[i][k] - nb5_l_x[3*i+k];
          vor_dist[-5+nb_cnt] += temp*temp;
        }
        vor_dist[-5+nb_cnt]=sqrt(vor_dist[-5+nb_cnt]);
      }
      // distance from neighbor +5
      if (me>NN-(5+1)) { vor_dist[5+nb_cnt] = lnhuge;
      }else{
        vor_dist[5+nb_cnt]=0.0;
        for(i=0;i<nlocal;i++)  if(Vid[i]==1) {
          k=0;
          temp = trial_x[i][k] - nb5_r_x[3*i+k];
          vor_dist[5+nb_cnt] += temp*temp;
          k=1;
          temp = trial_x[i][k] - nb5_r_x[3*i+k];
          vor_dist[5+nb_cnt] += temp*temp;
          k=2;
          temp = trial_x[i][k] - nb5_r_x[3*i+k]; 
          vor_dist[5+nb_cnt] += temp*temp; 
        }
        vor_dist[5+nb_cnt]=sqrt(vor_dist[5+nb_cnt]);
      }

      // distance from neighbor -4
      if (me<4) { vor_dist[-4+nb_cnt] = lnhuge;
      }else{
        vor_dist[-4+nb_cnt]=0.0;
        for(i=0;i<nlocal;i++)  if(Vid[i]==1) {
          k=0;
          temp = trial_x[i][k] - nb4_l_x[3*i+k];
          vor_dist[-4+nb_cnt] += temp*temp;
          k=1;
          temp = trial_x[i][k] - nb4_l_x[3*i+k];
          vor_dist[-4+nb_cnt] += temp*temp;
          k=2;
          temp = trial_x[i][k] - nb4_l_x[3*i+k]; 
          vor_dist[-4+nb_cnt] += temp*temp;
        }
        vor_dist[-4+nb_cnt]=sqrt(vor_dist[-4+nb_cnt]);
      }
      // distance from neighbor +4
      if (me>NN-(4+1)) { vor_dist[4+nb_cnt] = lnhuge;
      }else{
        vor_dist[4+nb_cnt]=0.0;
        for(i=0;i<nlocal;i++)  if(Vid[i]==1) {
          k=0;
          temp = trial_x[i][k] - nb4_r_x[3*i+k];
          vor_dist[4+nb_cnt] += temp*temp;
          k=1;
          temp = trial_x[i][k] - nb4_r_x[3*i+k];
          vor_dist[4+nb_cnt] += temp*temp;
          k=2; 
          temp = trial_x[i][k] - nb4_r_x[3*i+k]; 
          vor_dist[4+nb_cnt] += temp*temp;
        }
        vor_dist[4+nb_cnt]=sqrt(vor_dist[4+nb_cnt]);
      }

      // distance from neighbor -3
      if (me<3) { vor_dist[-3+nb_cnt] = lnhuge;
      }else{
        vor_dist[-3+nb_cnt]=0.0;
        for(i=0;i<nlocal;i++)  if(Vid[i]==1) {
          k=0; 
          temp = trial_x[i][k] - nb3_l_x[3*i+k];
          vor_dist[-3+nb_cnt] += temp*temp;
          k=1;
          temp = trial_x[i][k] - nb3_l_x[3*i+k];
          vor_dist[-3+nb_cnt] += temp*temp;
          k=2;
          temp = trial_x[i][k] - nb3_l_x[3*i+k]; 
          vor_dist[-3+nb_cnt] += temp*temp;
        }
        vor_dist[-3+nb_cnt]=sqrt(vor_dist[-3+nb_cnt]);
      }
      // distance from neighbor +3
      if (me>NN-(3+1)) { vor_dist[3+nb_cnt] = lnhuge;
      }else{
        vor_dist[3+nb_cnt]=0.0;
        for(i=0;i<nlocal;i++)  if(Vid[i]==1) {
          k=0;
          temp = trial_x[i][k] - nb3_r_x[3*i+k];
          vor_dist[3+nb_cnt] += temp*temp;
          k=1;
          temp = trial_x[i][k] - nb3_r_x[3*i+k];
          vor_dist[3+nb_cnt] += temp*temp;
          k=2;
          temp = trial_x[i][k] - nb3_r_x[3*i+k]; 
          vor_dist[3+nb_cnt] += temp*temp;
        }
        vor_dist[3+nb_cnt]=sqrt(vor_dist[3+nb_cnt]);
      }

      // distance from neighbor -2
      if (me<2) { vor_dist[-2+nb_cnt] = lnhuge;
      }else{
        vor_dist[-2+nb_cnt]=0.0;
        for(i=0;i<nlocal;i++)  if(Vid[i]==1) {
          k=0;
          temp = trial_x[i][k] - nb2_l_x[3*i+k];
          vor_dist[-2+nb_cnt] += temp*temp;
          k=1;
          temp = trial_x[i][k] - nb2_l_x[3*i+k];
          vor_dist[-2+nb_cnt] += temp*temp;
          k=2;
          temp = trial_x[i][k] - nb2_l_x[3*i+k]; 
          vor_dist[-2+nb_cnt] += temp*temp;
        }
        vor_dist[-2+nb_cnt]=sqrt(vor_dist[-2+nb_cnt]);
      }
      // distance from neighbor +2
      if (me>NN-(2+1)) { vor_dist[2+nb_cnt] = lnhuge;
      }else{
        vor_dist[2+nb_cnt]=0.0;
        for(i=0;i<nlocal;i++)  if(Vid[i]==1) {
          k=0;
          temp = trial_x[i][k] - nb2_r_x[3*i+k];
          vor_dist[2+nb_cnt] += temp*temp;
          k=1;
          temp = trial_x[i][k] - nb2_r_x[3*i+k];
          vor_dist[2+nb_cnt] += temp*temp;
          k=2; 
          temp = trial_x[i][k] - nb2_r_x[3*i+k]; 
          vor_dist[2+nb_cnt] += temp*temp;
        }
        vor_dist[2+nb_cnt]=sqrt(vor_dist[2+nb_cnt]);
      }

      // distance from neighbor -1
      if (me<1) { vor_dist[-1+nb_cnt] = lnhuge;
      }else{
        vor_dist[-1+nb_cnt]=0.0;
        for(i=0;i<nlocal;i++)  if(Vid[i]==1) {
          k=0;
          temp = trial_x[i][k] - nb1_l_x[3*i+k];
          vor_dist[-1+nb_cnt] += temp*temp;
          k=1;
          temp = trial_x[i][k] - nb1_l_x[3*i+k];
          vor_dist[-1+nb_cnt] += temp*temp;
          k=2;
          temp = trial_x[i][k] - nb1_l_x[3*i+k]; 
          vor_dist[-1+nb_cnt] += temp*temp;
        }
        vor_dist[-1+nb_cnt]=sqrt(vor_dist[-1+nb_cnt]);
      }
      // distance from neighbor +1
      if (me>NN-(1+1)) { vor_dist[1+nb_cnt] = lnhuge;
      }else{
        vor_dist[1+nb_cnt]=0.0;
        for(i=0;i<nlocal;i++)  if(Vid[i]==1) {
          k=0;
          temp = trial_x[i][k] - nb1_r_x[3*i+k];
          vor_dist[1+nb_cnt] += temp*temp;
          k=1;
          temp = trial_x[i][k] - nb1_r_x[3*i+k];
          vor_dist[1+nb_cnt] += temp*temp;
          k=2; 
          temp = trial_x[i][k] - nb1_r_x[3*i+k]; 
          vor_dist[1+nb_cnt] += temp*temp;
        }
        vor_dist[1+nb_cnt]=sqrt(vor_dist[1+nb_cnt]);
      }


    // find minimum distance to check with Voronoid cells
    vor_check = vor_dist[0];
    for(m=1;m<(2*nb_cnt+1);m++)  vor_check = MIN(vor_check,vor_dist[m]);
    if (fabs(vor_dist[0+nb_cnt]-vor_check)<1.0e-30) { for(i=0;i<nlocal;i++) for(k=0;k<3;k++) uw_x[i][k]=trial_x[i][k];
    }else{ for(i=0;i<nlocal;i++) if(Vid[i]!=1) for(k=0;k<3;k++) uw_x[i][k]=trial_x[i][k]; }

//    if (fabs(vor_dist[0+nb_cnt]-vor_check)<1.0e-30) {
//      for(i=0;i<nlocal;i++) for(k=0;k<3;k++) uw_x[i][k]=trial_x[i][k];
//      for(i=0;i<nlocal;i++) {
//        if(Vid[i]!=1) { for(k=0;k<3;k++) uw_x[i][k]=trial_x[i][k];
//        }else{ uw_x[i][1]=trial_x[i][1]; }
//      }
//    }


    if((pmf_call_counter)%1000==0 && me==0) cout<<"Step: "<<pmf_call_counter<<endl;


    /* Step 2: Update ave_x */

//    if (pmf_call_counter%reset_ave==0 && pmf_call_counter>reset_ave) {                  // start reset at 20000
//      for(i=0;i<nlocal;i++) for(k=0;k<3;k++)   ave_x[i][k] = str_x[3*i+k];
//      ave_counter += reset_ave;
//    }

    temp_step = static_cast<double>(pmf_call_counter);
    for(i=0;i<nlocal;i++) for(k=0;k<3;k++) ave_x[i][k] = (temp_step*ave_x[i][k] + uw_x[i][k])/(temp_step+1.0);


    /* Step 3: Propose trial images */
    kn=0.1*(static_cast<double>(NN))*0.1;

    // create trial images
    if (me==0) {
      for(i=0;i<nlocal;i++) {
        for(k=0;k<3;k++)   trial_str_x[3*i+k] = orig_x[i][k];                                                  // fix state 0
      }
    }else if(me>NN-2) {
      for(i=0;i<nlocal;i++) {
        if(Vid[i]==1) { for(k=0;k<3;k++)   trial_str_x[3*i+k] = str_x[3*i+k] - delta_tao*(str_x[3*i+k]-ave_x[i][k]);
        }else{ for(k=0;k<3;k++)    trial_str_x[3*i+k] = ave_x[i][k]; }
      }
    }else{
      for(i=0;i<nlocal;i++) {
        if(Vid[i]==1) {
          for(k=0;k<3;k++) {
            temp = kn*(nb1_r_x[3*i+k] + nb1_l_x[3*i+k] - 2.0*str_x[3*i+k]);
            trial_str_x[3*i+k] = str_x[3*i+k] - delta_tao*(str_x[3*i+k]-ave_x[i][k]) + temp;
          }
        }else{
          for(k=0;k<3;k++)    trial_str_x[3*i+k] = ave_x[i][k];
        }
      }
    }
    delete [] nb1_l_x;
    delete [] nb2_l_x;
    delete [] nb3_l_x;
    delete [] nb4_l_x;
    delete [] nb5_l_x;
    delete [] nb6_l_x;
    delete [] nb1_r_x;
    delete [] nb2_r_x;
    delete [] nb3_r_x;
    delete [] nb4_r_x;
    delete [] nb5_r_x;
    delete [] nb6_r_x;

    /* calculate image spacing and string length*/
    nb5_str_x = new double [3*nlocal];
    MPI_Barrier (universe->uworld);
    if (me==0) {
      MPI_Send(trial_str_x, 3*nlocal, MPI_DOUBLE, me+1, me+1, universe->uworld);
      img_spacing=0.0;
    }else{
      if (me==NN-1) MPI_Recv(nb5_str_x, 3*nlocal, MPI_DOUBLE, me-1, me, universe->uworld, &status);
      else MPI_Sendrecv(trial_str_x, 3*nlocal, MPI_DOUBLE, me+1, me+1, nb5_str_x, 3*nlocal, MPI_DOUBLE, me-1, me, universe->uworld, &status);    // send to the right, receive from the left
      img_spacing=0.0;
      for(i=0;i<nlocal;i++) if(Vid[i]==1) {
        k=0;
        temp = trial_str_x[3*i+k] - nb5_str_x[3*i+k];
        img_spacing += temp*temp;
        k=1;
        temp = trial_str_x[3*i+k] - nb5_str_x[3*i+k];
        img_spacing += temp*temp;
        k=2;
        temp = trial_str_x[3*i+k] - nb5_str_x[3*i+k];
        img_spacing += temp*temp;
      }
      img_spacing = sqrt(img_spacing);
    }
    MPI_Barrier (universe->uworld);
    MPI_Allgather(&img_spacing, 1, MPI_DOUBLE, g1_spacing, 1, MPI_DOUBLE, universe->uworld);
    length_along_line[0]=0.0;
    for(j=1;j<NN;j++)   length_along_line[j] = g1_spacing[j] + length_along_line[j-1];
    string_length=length_along_line[NN-1];

    /* print out image spacing and string length*/
    if (me==0) {
      if(pmf_call_counter%dump_spacing==0) {
        cout<<endl;
        cout<<"Step: "<<pmf_call_counter<<endl;
        cout<<"String length: "<<string_length/orig_string_length<<"     real length: "<<string_length<<endl;
        for(j=0;j<NN;j++)  cout<<j<<"  image spacing: "<<g1_spacing[j]<<endl;
      }
    }

    int temp_N;
    if (orig_string_length<string_length) {
      min_length = orig_string_length;
      temp_N = NN;
    }else {
      min_length = string_length;
      temp_N = NN-1;
    }


    /* Step 4: redistribute images along the piecewise linear curve */
//    if (me==0 || me==NN-1) { lj_marker=me;
    if (me==0) { lj_marker=me;                                                                                    //
    } else {
      if (me==NN-1 && temp_N==NN-1)   lj = min_length - 1.0e-10;
      else   lj = (static_cast<double>(me))*min_length/(static_cast<double>(NN-1));                               //
      for(m=1;m<NN;m++) if (length_along_line[m-1]<lj && lj<=length_along_line[m])    lj_marker=m;
    }
    MPI_Barrier (universe->uworld);
    MPI_Allgather(&lj_marker, 1, MPI_INT, g_marker, 1, MPI_INT, universe->uworld);


    // swap data for lj
    nb_lj1_x = new double [3*nlocal];
    nb_lj0_x = new double [3*nlocal];
    MPI_Barrier (universe->uworld);
    if (me!=g_marker[me]) {
      if (me==g_marker[me]-1) {
        MPI_Recv(nb_lj0_x, 3*nlocal, MPI_DOUBLE, g_marker[me], 0, universe->uworld, &status);
      }else{
        MPI_Recv(nb_lj0_x, 3*nlocal, MPI_DOUBLE, g_marker[me], 0, universe->uworld, &status);
        MPI_Recv(nb_lj1_x, 3*nlocal, MPI_DOUBLE, g_marker[me]-1, 1, universe->uworld, &status);
      }
    }
    for (j=1;j<temp_N;j++) {                                                                            // before NN-1 for without constraint
      if (j!=g_marker[j]) {
        if (j==g_marker[j]-1) {
          if (me==g_marker[j])     MPI_Send(trial_str_x, 3*nlocal, MPI_DOUBLE, j, 0, universe->uworld);
        }else{
          if (me==g_marker[j])     MPI_Send(trial_str_x, 3*nlocal, MPI_DOUBLE, j, 0, universe->uworld);
          if (me==g_marker[j]-1)   MPI_Send(trial_str_x, 3*nlocal, MPI_DOUBLE, j, 1, universe->uworld);
        }
      }
    }

    // adjust the images
//    if (me==0 || me==NN-1)  { for(i=0;i<nlocal;i++) for(k=0;k<3;k++)   str_x[3*i+k] = trial_str_x[3*i+k];
#if 1
    if (me==0)  { for(i=0;i<nlocal;i++) if(Vid[i]==1) for(k=0;k<3;k++)   str_x[3*i+k] = trial_str_x[3*i+k];                     //
    }else if(me==g_marker[me]) {
      for(i=0;i<nlocal;i++) {
        if(Vid[i]==1) {
          k=0;
          temp = ( trial_str_x[3*i+k] - nb5_str_x[3*i+k] ) / g1_spacing[lj_marker];
          str_x[3*i+k] = nb5_str_x[3*i+k] + temp*(lj-length_along_line[lj_marker-1]);
          k=1;
          temp = ( trial_str_x[3*i+k] - nb5_str_x[3*i+k] ) / g1_spacing[lj_marker];
          str_x[3*i+k] = nb5_str_x[3*i+k] + temp*(lj-length_along_line[lj_marker-1]);
          k=2;
          temp = ( trial_str_x[3*i+k] - nb5_str_x[3*i+k] ) / g1_spacing[lj_marker];
          str_x[3*i+k] = nb5_str_x[3*i+k] + temp*(lj-length_along_line[lj_marker-1]);
        }else{
          for(k=0;k<3;k++)   str_x[3*i+k] = trial_str_x[3*i+k];
        }
      }
    }else if(me==g_marker[me]-1) {
      for(i=0;i<nlocal;i++) {
        if(Vid[i]==1) {
          k=0;
          temp = ( nb_lj0_x[3*i+k] - trial_str_x[3*i+k] ) / g1_spacing[lj_marker];
          str_x[3*i+k] = trial_str_x[3*i+k] + temp*(lj-length_along_line[lj_marker-1]);
          k=1;
          temp = ( nb_lj0_x[3*i+k] - trial_str_x[3*i+k] ) / g1_spacing[lj_marker];
          str_x[3*i+k] = trial_str_x[3*i+k] + temp*(lj-length_along_line[lj_marker-1]);
          k=2;
          temp = ( nb_lj0_x[3*i+k] - trial_str_x[3*i+k] ) / g1_spacing[lj_marker];
          str_x[3*i+k] = trial_str_x[3*i+k] + temp*(lj-length_along_line[lj_marker-1]);
        }else{
          for(k=0;k<3;k++)   str_x[3*i+k] = trial_str_x[3*i+k];
        }
      }
    }else{
      for(i=0;i<nlocal;i++) {
        if(Vid[i]==1) {
          k=0;
          temp = ( nb_lj0_x[3*i+k] - nb_lj1_x[3*i+k] ) / g1_spacing[lj_marker];
          str_x[3*i+k] = nb_lj1_x[3*i+k] + temp*(lj-length_along_line[lj_marker-1]);
          k=1;
          temp = ( nb_lj0_x[3*i+k] - nb_lj1_x[3*i+k] ) / g1_spacing[lj_marker];
          str_x[3*i+k] = nb_lj1_x[3*i+k] + temp*(lj-length_along_line[lj_marker-1]);
          k=2;
          temp = ( nb_lj0_x[3*i+k] - nb_lj1_x[3*i+k] ) / g1_spacing[lj_marker];
          str_x[3*i+k] = nb_lj1_x[3*i+k] + temp*(lj-length_along_line[lj_marker-1]);
        }else{
          for(k=0;k<3;k++)   str_x[3*i+k] = trial_str_x[3*i+k];
        }
      }
    }
#endif
#if 0
    if (me==0)  { for(i=0;i<nlocal;i++) for(k=0;k<3;k++)   str_x[3*i+k] = trial_str_x[3*i+k];                     //
    }else if(me==g_marker[me]) {
      for(i=0;i<nlocal;i++) {
        for(k=0;k<3;k++) {
          temp = ( trial_str_x[3*i+k] - nb5_str_x[3*i+k] ) / g1_spacing[lj_marker];
          str_x[3*i+k] = nb5_str_x[3*i+k] + temp*(lj-length_along_line[lj_marker-1]);
        }
      }
    }else if(me==g_marker[me]-1) {
      for(i=0;i<nlocal;i++) {
        for(k=0;k<3;k++) {
          temp = ( nb_lj0_x[3*i+k] - trial_str_x[3*i+k] ) / g1_spacing[lj_marker];
          str_x[3*i+k] = trial_str_x[3*i+k] + temp*(lj-length_along_line[lj_marker-1]);
        }
      }
    }else{
      for(i=0;i<nlocal;i++) {
        for(k=0;k<3;k++) {
          temp = ( nb_lj0_x[3*i+k] - nb_lj1_x[3*i+k] ) / g1_spacing[lj_marker];
          str_x[3*i+k] = nb_lj1_x[3*i+k] + temp*(lj-length_along_line[lj_marker-1]);
        }
      }
    }
#endif
    delete [] nb5_str_x;
    delete [] nb_lj1_x;
    delete [] nb_lj0_x;


    /* calculate image spacing after adjustment */ 
    if(pmf_call_counter%dump_spacing==0) {
      nb6_str_x = new double [3*nlocal];
      MPI_Barrier (universe->uworld);
      if (me<1) {
        MPI_Send(str_x, 3*nlocal, MPI_DOUBLE, me+1, me+1, universe->uworld);
        img_spacing=0.0;
      }else{
        if (me>NN-2) MPI_Recv(nb6_str_x, 3*nlocal, MPI_DOUBLE, me-1, me, universe->uworld, &status);
        else MPI_Sendrecv(str_x, 3*nlocal, MPI_DOUBLE, me+1, me+1, nb6_str_x, 3*nlocal, MPI_DOUBLE, me-1, me, universe->uworld, &status);    // send to +1, receive from -1
        img_spacing=0.0;
        for(i=0;i<nlocal;i++) if(Vid[i]==1) {
          k=0;
          temp = str_x[3*i+k] - nb6_str_x[3*i+k];
          img_spacing += temp*temp;
          k=1;
          temp = str_x[3*i+k] - nb6_str_x[3*i+k];
          img_spacing += temp*temp;
          k=2;
          temp = str_x[3*i+k] - nb6_str_x[3*i+k];
          img_spacing += temp*temp;
        }
        img_spacing = sqrt(img_spacing);
      }

      MPI_Barrier (universe->uworld);
      MPI_Allgather(&img_spacing, 1, MPI_DOUBLE, g2_spacing, 1, MPI_DOUBLE, universe->uworld);
      length_along_line[0]=0.0;
      for(j=1;j<NN;j++)   length_along_line[j] = g2_spacing[j] + length_along_line[j-1];
      string_length=length_along_line[NN-1];

      // print the New string length --- before convergence
      if (me==0) {
        cout<<"New string length: "<<string_length/orig_string_length<<"     real length: "<<string_length<<endl;
        for(j=0;j<NN;j++)  cout<<j<<"  New image spacing: "<<g2_spacing[j]<<endl;
      }
      delete [] nb6_str_x;
    }


    #if 0
    // print the position of image 0
    if(pmf_call_counter%dump_spacing==0) {
      if (me==0) {
        double anchor0=0.0;
        for(i=0;i<nlocal;i++) for(k=0;k<3;k++)   { temp = str_x[3*i+k] - orig_x[i][k]; anchor0 += temp*temp; }
        anchor0=sqrt(anchor0);
        cout<<"anchor0: "<<anchor0<<endl;
        cout<<endl;
      }
    }
    #endif



    /* Step 5: check with NEW Voronoi cell to verify the updated uw_x */

      for(m=0;m<(2*nb_cnt+1);m++)   vor_dist[m]=0.0;

      /* swaping new string */
      // send to +6; receive from -6
      nb6_l_x = new double [3*nlocal];
      MPI_Barrier (universe->uworld);
      if (me<6) { 
        MPI_Send(str_x, 3*nlocal, MPI_DOUBLE, me+6, me+6, universe->uworld);
      }else{
        if (me>NN-(6+1)) MPI_Recv(nb6_l_x, 3*nlocal, MPI_DOUBLE, me-6, me, universe->uworld, &status);
        else MPI_Sendrecv(str_x, 3*nlocal, MPI_DOUBLE, me+6, me+6, nb6_l_x, 3*nlocal, MPI_DOUBLE, me-6, me, universe->uworld, &status);
      }
      // send to -6; receive from +6
      nb6_r_x = new double [3*nlocal];
      MPI_Barrier (universe->uworld);
      if (me>NN-(6+1)) {
        MPI_Send(str_x, 3*nlocal, MPI_DOUBLE, me-6, me-6, universe->uworld);
      }else{
        if (me<6) MPI_Recv(nb6_r_x, 3*nlocal, MPI_DOUBLE, me+6, me, universe->uworld, &status);
        else MPI_Sendrecv(str_x, 3*nlocal, MPI_DOUBLE, me-6, me-6, nb6_r_x, 3*nlocal, MPI_DOUBLE, me+6, me, universe->uworld, &status);
      }

      // send to +5; receive from -5
      nb5_l_x = new double [3*nlocal];
      MPI_Barrier (universe->uworld);
      if (me<5) {
        MPI_Send(str_x, 3*nlocal, MPI_DOUBLE, me+5, me+5, universe->uworld);
      }else{
        if (me>NN-(5+1)) MPI_Recv(nb5_l_x, 3*nlocal, MPI_DOUBLE, me-5, me, universe->uworld, &status);
        else MPI_Sendrecv(str_x, 3*nlocal, MPI_DOUBLE, me+5, me+5, nb5_l_x, 3*nlocal, MPI_DOUBLE, me-5, me, universe->uworld, &status);
      }
      // send to -5; receive from +5
      nb5_r_x = new double [3*nlocal];
      MPI_Barrier (universe->uworld);
      if (me>NN-(5+1)) {
        MPI_Send(str_x, 3*nlocal, MPI_DOUBLE, me-5, me-5, universe->uworld);
      }else{
        if (me<5) MPI_Recv(nb5_r_x, 3*nlocal, MPI_DOUBLE, me+5, me, universe->uworld, &status);
        else MPI_Sendrecv(str_x, 3*nlocal, MPI_DOUBLE, me-5, me-5, nb5_r_x, 3*nlocal, MPI_DOUBLE, me+5, me, universe->uworld, &status);
      }

      // send to +4; receive from -4
      nb4_l_x = new double [3*nlocal];
      MPI_Barrier (universe->uworld);
      if (me<4) {
        MPI_Send(str_x, 3*nlocal, MPI_DOUBLE, me+4, me+4, universe->uworld);
      }else{
        if (me>NN-(4+1)) MPI_Recv(nb4_l_x, 3*nlocal, MPI_DOUBLE, me-4, me, universe->uworld, &status);
        else MPI_Sendrecv(str_x, 3*nlocal, MPI_DOUBLE, me+4, me+4, nb4_l_x, 3*nlocal, MPI_DOUBLE, me-4, me, universe->uworld, &status);
      }
      // send to -4; receive from +4
      nb4_r_x = new double [3*nlocal];
      MPI_Barrier (universe->uworld);
      if (me>NN-(4+1)) {
        MPI_Send(str_x, 3*nlocal, MPI_DOUBLE, me-4, me-4, universe->uworld);
      }else{
        if (me<4) MPI_Recv(nb4_r_x, 3*nlocal, MPI_DOUBLE, me+4, me, universe->uworld, &status);
        else MPI_Sendrecv(str_x, 3*nlocal, MPI_DOUBLE, me-4, me-4, nb4_r_x, 3*nlocal, MPI_DOUBLE, me+4, me, universe->uworld, &status);
      }
      // send to +3; receive from -3
      nb3_l_x = new double [3*nlocal];
      MPI_Barrier (universe->uworld);
      if (me<3) {
        MPI_Send(str_x, 3*nlocal, MPI_DOUBLE, me+3, me+3, universe->uworld);
      }else{
        if (me>NN-(3+1)) MPI_Recv(nb3_l_x, 3*nlocal, MPI_DOUBLE, me-3, me, universe->uworld, &status);
        else MPI_Sendrecv(str_x, 3*nlocal, MPI_DOUBLE, me+3, me+3, nb3_l_x, 3*nlocal, MPI_DOUBLE, me-3, me, universe->uworld, &status);
      }
      // send to -3; receive from +3
      nb3_r_x = new double [3*nlocal];
      MPI_Barrier (universe->uworld);
      if (me>NN-(3+1)) {
        MPI_Send(str_x, 3*nlocal, MPI_DOUBLE, me-3, me-3, universe->uworld);
      }else{
        if (me<3) MPI_Recv(nb3_r_x, 3*nlocal, MPI_DOUBLE, me+3, me, universe->uworld, &status);
        else MPI_Sendrecv(str_x, 3*nlocal, MPI_DOUBLE, me-3, me-3, nb3_r_x, 3*nlocal, MPI_DOUBLE, me+3, me, universe->uworld, &status);
      }
      // send to +2; receive from -2
      nb2_l_x = new double [3*nlocal];
      MPI_Barrier (universe->uworld);
      if (me<2) {
        MPI_Send(str_x, 3*nlocal, MPI_DOUBLE, me+2, me+2, universe->uworld);
      }else{
        if (me>NN-(2+1)) MPI_Recv(nb2_l_x, 3*nlocal, MPI_DOUBLE, me-2, me, universe->uworld, &status);
        else MPI_Sendrecv(str_x, 3*nlocal, MPI_DOUBLE, me+2, me+2, nb2_l_x, 3*nlocal, MPI_DOUBLE, me-2, me, universe->uworld, &status);
      }
      // send to -2; receive from +2
      nb2_r_x = new double [3*nlocal];
      MPI_Barrier (universe->uworld);
      if (me>NN-(2+1)) {
        MPI_Send(str_x, 3*nlocal, MPI_DOUBLE, me-2, me-2, universe->uworld);
      }else{
        if (me<2) MPI_Recv(nb2_r_x, 3*nlocal, MPI_DOUBLE, me+2, me, universe->uworld, &status);
        else MPI_Sendrecv(str_x, 3*nlocal, MPI_DOUBLE, me-2, me-2, nb2_r_x, 3*nlocal, MPI_DOUBLE, me+2, me, universe->uworld, &status);
      }
      // send to +1; receive from -1
      nb1_l_x = new double [3*nlocal];
      MPI_Barrier (universe->uworld);
      if (me<1) {
        MPI_Send(str_x, 3*nlocal, MPI_DOUBLE, me+1, me+1, universe->uworld);
      }else{
        if (me>NN-(1+1)) MPI_Recv(nb1_l_x, 3*nlocal, MPI_DOUBLE, me-1, me, universe->uworld, &status);
        else MPI_Sendrecv(str_x, 3*nlocal, MPI_DOUBLE, me+1, me+1, nb1_l_x, 3*nlocal, MPI_DOUBLE, me-1, me, universe->uworld, &status);
      }
      // send to -1; receive from +1
      nb1_r_x = new double [3*nlocal];
      MPI_Barrier (universe->uworld);
      if (me>NN-(1+1)) {
        MPI_Send(str_x, 3*nlocal, MPI_DOUBLE, me-1, me-1, universe->uworld);
      }else{
        if (me<1) MPI_Recv(nb1_r_x, 3*nlocal, MPI_DOUBLE, me+1, me, universe->uworld, &status);
        else MPI_Sendrecv(str_x, 3*nlocal, MPI_DOUBLE, me-1, me-1, nb1_r_x, 3*nlocal, MPI_DOUBLE, me+1, me, universe->uworld, &status);
      }


      /* distance with new Voronoid cells */

      // with center cell
      vor_dist[0+nb_cnt]=0.0;
      for(i=0;i<nlocal;i++) if(Vid[i]==1) {
        k=0;
        temp = uw_x[i][k] - str_x[3*i+k];
        vor_dist[0+nb_cnt] += temp*temp;
        k=1;
        temp = uw_x[i][k] - str_x[3*i+k];
        vor_dist[0+nb_cnt] += temp*temp;
        k=2;
        temp = uw_x[i][k] - str_x[3*i+k]; 
        vor_dist[0+nb_cnt] += temp*temp;
      }
      vor_dist[0+nb_cnt]=sqrt(vor_dist[0+nb_cnt]);

      // distance from neighbor -6
      if (me<6) { vor_dist[-6+nb_cnt] = lnhuge;
      }else{
        vor_dist[-6+nb_cnt]=0.0;
        for(i=0;i<nlocal;i++)  if(Vid[i]==1) {
          k=0;
          temp = trial_x[i][k] - nb6_l_x[3*i+k];
          vor_dist[-6+nb_cnt] += temp*temp;
          k=1;
          temp = trial_x[i][k] - nb6_l_x[3*i+k];
          vor_dist[-6+nb_cnt] += temp*temp;
          k=2;
          temp = trial_x[i][k] - nb6_l_x[3*i+k];
          vor_dist[-6+nb_cnt] += temp*temp;
        }
        vor_dist[-6+nb_cnt]=sqrt(vor_dist[-6+nb_cnt]);
      }
      // distance from neighbor +6
      if (me>NN-(6+1)) { vor_dist[6+nb_cnt] = lnhuge;
      }else{
        vor_dist[6+nb_cnt]=0.0;
        for(i=0;i<nlocal;i++)  if(Vid[i]==1) {
          k=0;
          temp = trial_x[i][k] - nb6_r_x[3*i+k];
          vor_dist[6+nb_cnt] += temp*temp;
          k=1;
          temp = trial_x[i][k] - nb6_r_x[3*i+k];
          vor_dist[6+nb_cnt] += temp*temp;
          k=2;
          temp = trial_x[i][k] - nb6_r_x[3*i+k];
          vor_dist[6+nb_cnt] += temp*temp;
        }
        vor_dist[6+nb_cnt]=sqrt(vor_dist[6+nb_cnt]);
      }

      // distance from neighbor -5
      if (me<5) { vor_dist[-5+nb_cnt] = lnhuge;
      }else{
        vor_dist[-5+nb_cnt]=0.0;
        for(i=0;i<nlocal;i++)  if(Vid[i]==1) {
          k=0;
          temp = trial_x[i][k] - nb5_l_x[3*i+k];
          vor_dist[-5+nb_cnt] += temp*temp;
          k=1;
          temp = trial_x[i][k] - nb5_l_x[3*i+k];
          vor_dist[-5+nb_cnt] += temp*temp;
          k=2;
          temp = trial_x[i][k] - nb5_l_x[3*i+k];
          vor_dist[-5+nb_cnt] += temp*temp;
        }
        vor_dist[-5+nb_cnt]=sqrt(vor_dist[-5+nb_cnt]);
      }
      // distance from neighbor +5
      if (me>NN-(5+1)) { vor_dist[5+nb_cnt] = lnhuge;
      }else{
        vor_dist[5+nb_cnt]=0.0;
        for(i=0;i<nlocal;i++)  if(Vid[i]==1) {
          k=0;
          temp = trial_x[i][k] - nb5_r_x[3*i+k];
          vor_dist[5+nb_cnt] += temp*temp;
          k=1;
          temp = trial_x[i][k] - nb5_r_x[3*i+k];
          vor_dist[5+nb_cnt] += temp*temp;
          k=2;
          temp = trial_x[i][k] - nb5_r_x[3*i+k];
          vor_dist[5+nb_cnt] += temp*temp;
        }
        vor_dist[5+nb_cnt]=sqrt(vor_dist[5+nb_cnt]);
      }

      // distance from neighbor -4
      if (me<4) { vor_dist[-4+nb_cnt] = lnhuge;
      }else{
        vor_dist[-4+nb_cnt]=0.0;
        for(i=0;i<nlocal;i++)  if(Vid[i]==1) {
          k=0;
          temp = uw_x[i][k] - nb4_l_x[3*i+k];
          vor_dist[-4+nb_cnt] += temp*temp;
          k=1;
          temp = uw_x[i][k] - nb4_l_x[3*i+k];
          vor_dist[-4+nb_cnt] += temp*temp;
          k=2;
          temp = uw_x[i][k] - nb4_l_x[3*i+k]; 
          vor_dist[-4+nb_cnt] += temp*temp; 
        }
        vor_dist[-4+nb_cnt]=sqrt(vor_dist[-4+nb_cnt]);
      }
      // distance from neighbor +4
      if (me>NN-(4+1)) { vor_dist[4+nb_cnt] = lnhuge;
      }else{
        vor_dist[4+nb_cnt]=0.0;
        for(i=0;i<nlocal;i++)  if(Vid[i]==1) {
          k=0;
          temp = uw_x[i][k] - nb4_r_x[3*i+k];
          vor_dist[4+nb_cnt] += temp*temp;
          k=1;
          temp = uw_x[i][k] - nb4_r_x[3*i+k];
          vor_dist[4+nb_cnt] += temp*temp;
          k=2; 
          temp = uw_x[i][k] - nb4_r_x[3*i+k]; 
          vor_dist[4+nb_cnt] += temp*temp; 
        }
        vor_dist[4+nb_cnt]=sqrt(vor_dist[4+nb_cnt]);
      }
      // distance from neighbor -3
      if (me<3) { vor_dist[-3+nb_cnt] = lnhuge;
      }else{
        vor_dist[-3+nb_cnt]=0.0;
        for(i=0;i<nlocal;i++)  if(Vid[i]==1) {
          k=0;
          temp = uw_x[i][k] - nb3_l_x[3*i+k];
          vor_dist[-3+nb_cnt] += temp*temp;
          k=1;
          temp = uw_x[i][k] - nb3_l_x[3*i+k];
          vor_dist[-3+nb_cnt] += temp*temp;
          k=2; 
          temp = uw_x[i][k] - nb3_l_x[3*i+k]; 
          vor_dist[-3+nb_cnt] += temp*temp; 
        }
        vor_dist[-3+nb_cnt]=sqrt(vor_dist[-3+nb_cnt]);
      }
      // distance from neighbor +3
      if (me>NN-(3+1)) { vor_dist[3+nb_cnt] = lnhuge;
      }else{
        vor_dist[3+nb_cnt]=0.0;
        for(i=0;i<nlocal;i++)  if(Vid[i]==1) {
          k=0;
          temp = uw_x[i][k] - nb3_r_x[3*i+k];
          vor_dist[3+nb_cnt] += temp*temp;
          k=1;
          temp = uw_x[i][k] - nb3_r_x[3*i+k];
          vor_dist[3+nb_cnt] += temp*temp;
          k=2;
          temp = uw_x[i][k] - nb3_r_x[3*i+k]; 
          vor_dist[3+nb_cnt] += temp*temp; 
        }
        vor_dist[3+nb_cnt]=sqrt(vor_dist[3+nb_cnt]);
      }
      // distance from neighbor -2
      if (me<2) { vor_dist[-2+nb_cnt] = lnhuge;
      }else{
        vor_dist[-2+nb_cnt]=0.0;
        for(i=0;i<nlocal;i++)  if(Vid[i]==1) {
          k=0;
          temp = uw_x[i][k] - nb2_l_x[3*i+k];
          vor_dist[-2+nb_cnt] += temp*temp;
          k=1;
          temp = uw_x[i][k] - nb2_l_x[3*i+k];
          vor_dist[-2+nb_cnt] += temp*temp;
          k=2;
          temp = uw_x[i][k] - nb2_l_x[3*i+k]; 
          vor_dist[-2+nb_cnt] += temp*temp; 
        }
        vor_dist[-2+nb_cnt]=sqrt(vor_dist[-2+nb_cnt]);
      }
      // distance from neighbor +2
      if (me>NN-(2+1)) { vor_dist[2+nb_cnt] = lnhuge;
      }else{
        vor_dist[2+nb_cnt]=0.0;
        for(i=0;i<nlocal;i++)  if(Vid[i]==1) {
          k=0;
          temp = uw_x[i][k] - nb2_r_x[3*i+k];
          vor_dist[2+nb_cnt] += temp*temp;
          k=1;
          temp = uw_x[i][k] - nb2_r_x[3*i+k];
          vor_dist[2+nb_cnt] += temp*temp;
          k=2;
          temp = uw_x[i][k] - nb2_r_x[3*i+k]; 
          vor_dist[2+nb_cnt] += temp*temp;
        }
        vor_dist[2+nb_cnt]=sqrt(vor_dist[2+nb_cnt]);
      }
      // distance from neighbor -1
      if (me<1) { vor_dist[-1+nb_cnt] = lnhuge;
      }else{
        vor_dist[-1+nb_cnt]=0.0;
        for(i=0;i<nlocal;i++)  if(Vid[i]==1) {
          k=0;
          temp = uw_x[i][k] - nb1_l_x[3*i+k];
          vor_dist[-1+nb_cnt] += temp*temp;
          k=1;
          temp = uw_x[i][k] - nb1_l_x[3*i+k];
          vor_dist[-1+nb_cnt] += temp*temp;
          k=2;
          temp = uw_x[i][k] - nb1_l_x[3*i+k]; 
          vor_dist[-1+nb_cnt] += temp*temp;
        }
        vor_dist[-1+nb_cnt]=sqrt(vor_dist[-1+nb_cnt]);
      }
      // distance from neighbor +1
      if (me>NN-(1+1)) { vor_dist[1+nb_cnt] = lnhuge;
      }else{
        vor_dist[1+nb_cnt]=0.0;
        for(i=0;i<nlocal;i++)  if(Vid[i]==1) {
          k=0;
          temp = uw_x[i][k] - nb1_r_x[3*i+k];
          vor_dist[1+nb_cnt] += temp*temp;
          k=1;
          temp = uw_x[i][k] - nb1_r_x[3*i+k];
          vor_dist[1+nb_cnt] += temp*temp;
          k=2;
          temp = uw_x[i][k] - nb1_r_x[3*i+k]; 
          vor_dist[1+nb_cnt] += temp*temp;
        }
        vor_dist[1+nb_cnt]=sqrt(vor_dist[1+nb_cnt]);
      }

#if 1
    if (pmf_call_counter>0) {
      // calculate PMF vector spacing
      if (me==0) {
        img_spacing=0.0;
        for(i=0;i<nlocal;i++) if(Vid[i]==1) for(k=0;k<3;k++)   { temp = nb1_r_x[3*i+k] - str_x[3*i+k]; img_spacing += temp*temp; }
        img_spacing = sqrt(img_spacing);
      }else if(me==NN-1) {
        img_spacing=0.0;
        for(i=0;i<nlocal;i++) if(Vid[i]==1) for(k=0;k<3;k++)   { temp = str_x[3*i+k] - nb1_l_x[3*i+k]; img_spacing += temp*temp; }
        img_spacing = sqrt(img_spacing);
      }else{
        img_spacing=0.0;
        for(i=0;i<nlocal;i++) if(Vid[i]==1) for(k=0;k<3;k++)   { temp = nb1_r_x[3*i+k] - nb1_l_x[3*i+k]; img_spacing += temp*temp; }
        img_spacing = sqrt(img_spacing);
      }
      // create normalized pmf vector
      if (me==0) {
        for(i=0;i<nlocal;i++) if(Vid[i]==1) for(k=0;k<3;k++)  { pmf_vec[3*i+k] = nb1_r_x[3*i+k] - str_x[3*i+k]; pmf_vec[3*i+k] /= img_spacing; }
      }else if(me==NN-1) {
        for(i=0;i<nlocal;i++) if(Vid[i]==1) for(k=0;k<3;k++)  { pmf_vec[3*i+k] = str_x[3*i+k] - nb1_l_x[3*i+k]; pmf_vec[3*i+k] /= img_spacing; }
      }else{
        for(i=0;i<nlocal;i++) if(Vid[i]==1) for(k=0;k<3;k++)  { pmf_vec[3*i+k] = nb1_r_x[3*i+k] - nb1_l_x[3*i+k]; pmf_vec[3*i+k] /= img_spacing; }
      }
    }
    // calculate and print mean force on string --- after convergence
    pforce_dot_prod=0.0;
    for(i=0;i<nlocal;i++) if(Vid[i]==1) for(k=0;k<3;k++)  { pforce_dot_prod += pmf_vec[3*i+k]*f[i][k]; }
    ave_pforce += pforce_dot_prod;

    MPI_Barrier (universe->uworld);
    if(pmf_call_counter%dump_force==0){
      temp_pforce = ave_pforce;
      temp_pforce /= (static_cast<double>(pmf_call_counter));
      MPI_Gather(&temp_pforce, 1, MPI_DOUBLE, g_ave_pforce, 1, MPI_DOUBLE, 0, universe->uworld);
      // printing
      if (me==0) {
        char buf[256];
        sprintf(buf,"mean-force_string_Vcell.%d",pmf_call_counter);
        ofstream outdata (buf);
        outdata<<"Step: "<<pmf_call_counter<<endl;
        for(j=0;j<NN;j++)  outdata<<g_ave_pforce[j]<<endl;
        outdata.close();
        for(j=0;j<NN;j++)  cout<<j<<"      pmf_force:  "<<g_ave_pforce[j]<<endl;
      }
    }
#endif
      delete [] nb1_l_x;
      delete [] nb2_l_x;
      delete [] nb3_l_x;
      delete [] nb4_l_x;
      delete [] nb5_l_x;
      delete [] nb6_l_x;
      delete [] nb1_r_x;
      delete [] nb2_r_x;
      delete [] nb3_r_x;
      delete [] nb4_r_x;
      delete [] nb5_r_x;
      delete [] nb6_r_x;


    // check with new adjusted Voronoid cells
    vor_check = vor_dist[0];
    for(m=1;m<(2*nb_cnt+1);m++)  vor_check = MIN(vor_check,vor_dist[m]);
    if (fabs(vor_dist[0+nb_cnt]-vor_check)>1.0e-30) {
      for(i=0;i<nlocal;i++)   if(Vid[i]==1)   for(k=0;k<3;k++)   uw_x[i][k]=str_x[3*i+k];
    }


    // reset state NN
    if (me==NN-1) {
      if (vor_dist[0+nb_cnt]>1.0) {
        for(i=0;i<nlocal;i++)   if(Vid[i]==1)   for(k=0;k<3;k++)   uw_x[i][k] = str_x[3*i+k];
      }
    }


    /* print images */
    if(pmf_call_counter%dump_img==0){
        char buf[256];
        sprintf(buf,"image_Vcell.%d.300K.%d", me+1, pmf_call_counter);
        ofstream outdata (buf);
        outdata<<"ITEM: TIMESTEP"<<endl;
        outdata<<pmf_call_counter<<endl;
        outdata<<"ITEM: NUMBER OF ATOMS"<<endl;
        outdata<<nlocal<<endl;
        outdata<<"ITEM: BOX BOUNDS x y z"<<endl;
        outdata<<setprecision(9)<<xlo<<" "<<xhi<<endl;
        outdata<<setprecision(9)<<ylo<<" "<<yhi<<endl;
        outdata<<setprecision(9)<<zlo<<" "<<zhi<<endl;
        outdata<<"ITEM: ATOMS id type x y z"<<endl;
        for(i=0;i<nlocal;i++)  outdata<<setprecision(12)<<i+1<<" "<<type[i]<<" "<<str_x[3*i+0]<<" "<<str_x[3*i+1]<<" "<<str_x[3*i+2]<<endl;
        outdata.close();
    }

  }

    /* ----------update wrapped coordinates used outside of this routine--------------- */

    /* allow both end states to move */
       for(i=0;i<nlocal;i++) for(k=0;k<3;k++)  x[i][k] += (uw_x[i][k]-old_x[i][k]);

}


//*************** I didn't change anything below this point -Kris
/* ---------------------------------------------------------------------- */
void FixNVE::final_integrate()
{
  double dtfm; 

  double **v = atom->v;
  double **f = atom->f; 
  double *mass = atom->mass;
  double *rmass = atom->rmass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal; 

  /* pmf variables */
  double f_scaler=0;
  static double pmf_printed=0;
  double tmag, tscaler;
  static double thermterm=0;
  int k, i;
   
  /* compute pmf force for step n+1 */
  dtfm = dtf / mass[type[0]];
 // pmf_force_scaler-=x_scaler/(dtfm*dtv*2);

if (pmf_call_counter>=MD_start) {
  if (rmass) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        dtfm = dtf / rmass[i];
        v[i][0] += dtfm * f[i][0];
        v[i][1] += dtfm * f[i][1];
        v[i][2] += dtfm * f[i][2];
      }
    }

  } else {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        dtfm = dtf / mass[type[i]];
        v[i][0] += dtfm * f[i][0];
        v[i][1] += dtfm * f[i][1];
        v[i][2] += dtfm * f[i][2];
      }
    }
  }
}

/* note I believe that the below commented section should be turned on for
temperatures greater than zero.  However, when turning it on we get unrealistic forces. Without, things seem to work OK.
General PMF theory makes me think it should be on while a targeted MD paper may suggest other wise */
#if 0
   /* compute temperature */
   tscaler=0.0;
   if (rmass) {
    for (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit)
        tscaler += (v[i][0]*v[i][0] + v[i][1]*v[i][1] + v[i][2]*v[i][2]) * rmass[i];
  } else {
    for (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit)
        tscaler += (v[i][0]*v[i][0] + v[i][1]*v[i][1] + v[i][2]*v[i][2]) *
          mass[type[i]];
  }
  tscaler*=1.0364269e-4/(3.0*8.617343e-5*nlocal);
  
   /* adjust pmf force for polare coordinates */
   thermterm-=tscaler*8.617343e-5/pmf_vec_length*(nlocal*3-4);
   pmf_force_scaler-=tscaler*8.617343e-5/pmf_vec_length*(nlocal*3-4);
#endif

  pmf_call_counter += 1;

}

/* ---------------------------------------------------------------------- */

void FixNVE::initial_integrate_respa(int vflag, int ilevel, int flag)
{
  if (flag) return;             // only used by NPT,NPH

  dtv = step_respa[ilevel];
  dtf = 0.5 * step_respa[ilevel] * force->ftm2v;

  if (ilevel == 0) initial_integrate(vflag);
  else final_integrate();
}

/* ---------------------------------------------------------------------- */

void FixNVE::final_integrate_respa(int ilevel)
{
  dtf = 0.5 * step_respa[ilevel] * force->ftm2v;
  final_integrate();
}

/* ---------------------------------------------------------------------- */

void FixNVE::reset_dt()
{
  dtv = update->dt;
  dtf = 0.5 * update->dt * force->ftm2v;
  //dtq = 0.5 * update->dt;
}
