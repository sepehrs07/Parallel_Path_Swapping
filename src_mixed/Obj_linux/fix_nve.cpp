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
#include "iostream"
#include "fstream"
#include "math.h"
#include "random_mars.h"
#include "iomanip"
#include "universe.h"
#include "compute.h"
#include "compute_temp.h"
#include "modify.h"
#include "group.h"
#include "domain.h"
#include <iostream>
#include <cstdlib>
#include <omp.h>

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

using namespace std;
using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

FixNVE::FixNVE(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (strcmp(style,"nve/sphere") != 0 && narg < 3)
    error->all("Illegal fix nve command");

  time_integrate = 1;
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
  dtf = 0.5 * update->dt * force->ftm2v;

  if (strcmp(update->integrate_style,"respa") == 0)
    step_respa = ((Respa *) update->integrate)->step;
}

/* ---------------------------------------------------------------------- */


 /* TIS global variables */
 static int    rescale_check=0;
 static int    g_step=0;
 static int    success_counter=0;
 static int    Tattempt_counter=-1; 
 static double xlo, xhi, ylo, yhi, zlo, zhi;
 static double BL[3], orig_com[3];

 static double       center_x[14000*3][50];

 static double      orig_x[14000][3];
 static double        uw_x[14000][3];
 static double       old_x[14000][3];
 static double         X_0[14000][3];
 static double         V_0[14000][3];
// static double    temp_X_0[20000][3];
// static double    temp_V_0[20000][3];
// static double   add_force[20000];
 static double   atom_mass[14000];
 static double   int_time = 0;
 static double   t_start;
 static double       st3_x[14000][3];
 static double       st3_v[14000][3];
 static double       old_v[14000][3];
// static double     trial_v[14000][3];

// static double   Al_mass = 26.982; //made same as potential file (dw)
 static double   H, PE_0, velo_magn, ke_added, cum_E;
 static int      tao_b, tao_f, step_bck, step_fwd, oldm, picked_m,N_max;
 static int      success_index, crossed_interface, previous_success, old_success_index;
 static int      unique_trajectories =0;
 static int      ensemble_trajectories = 0;
 const int      N_hist = 300;
 static int      count_hist = -1;
 static int     count_hist_0;
 static double   x_hist_0[14000][3][2*N_hist+2],x_hist[14000][3][2*N_hist+2],v_hist_0[14000][3][2*N_hist+2],v_hist[14000][3][2*N_hist+2];
/* double***       x_hist_0;
 double***       x_hist;
 double***       v_hist_0;
 double***       v_hist;*/
// static int      flag_1st_traj = 0;
// static int      cell_pick, cell_s2p=3;

 static int      b_index=0;
 static int      f_index=0;
 //static int      XV0_rec=0;
 static int      vst3_rec=0;
 static int      pick_index=0;
 static int      step6_index=0;
 static int      st3_pass=0;
 static int      trajectory_finished=1;
 static int      time_L_old = 10000;
 static int      rsteps=100;

 static int      NN; // = 11;                  //number of cells from cell #0 to cell #NN-1
// static int      cell_ground = 2;
 static int      cell_start;
 static int      lamda;          // last cell # (from 0 to NN-1) (trajectory stops when it touches this cell)

/* ----------------------------------------------------------------------
   allow for both per-type and per-atom mass
------------------------------------------------------------------------- */

void FixNVE::initial_integrate(int vflag)
{ 
  #pragma omp barrier
 /* TIS variables */
  double   old_ke, new_ke, ave_E, sample_mass, success_rate, TIS_prob;
  double   rand_ensemble,rand_U, rand_U1, rand_U2, rand_G, prob, temp, trash, vor_check;
  double   vor_dist[NN];
  int ranseed;
  int *ranseed_pointer;
  int      rescale=0;
  int pnum;
  /*---------------*/

  double dtfm;

  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double *mass = atom->mass;
  double *rmass = atom->rmass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  //MPI_Status status;
  //int me;
  //MPI_Comm_rank(universe->uworld, &me);


  /* Initialization for first call to fix_nve */
    if(g_step==0) {
     /* #pragma omp parallel
      {
      cout<<"number of threads is: "<<omp_get_num_threads()<<endl;
      }*/
      t_start = omp_get_wtime();
      /* read if constraint is on/off and box sizes */
      ifstream vector("pos.read");
      vector>>xlo>>xhi;
      vector>>ylo>>yhi;
      vector>>zlo>>zhi;
//      vector>>lamda;
      vector>>sample_mass;
      for(int i=0;i<nlocal;i++)  vector>>trash>>atom_mass[i]>>trash;      // for adding force
      vector.close();
      BL[0] = xhi-xlo;
      BL[1] = yhi-ylo;
      BL[2] = zhi-zlo;

      /* read interface position */
      ifstream inld("ld_run.in");
      inld>>lamda;
      inld.close();
      
      ifstream inld1("seed.in");
      inld1>>ranseed;
      inld1.close();
      srand(ranseed);

      char buf1[128];
      sprintf(buf1,"cp ld%d.in ld_run.in",lamda+1);
      system (buf1);

      NN = lamda+2+3;
      cell_start = lamda+1;

      /* read PE at beginning and Interface*/
      char buf0[128];
      sprintf(buf0,"PE.log");
      ifstream indata0 (buf0);
      for(int i=0;i<cell_start;i++)  indata0>>PE_0;
      indata0.close();
      cout<<"from interface #"<<lamda<<"  to #"<<lamda+1<<endl;

      // read starting configuration 
      char bufdt[128];
      sprintf(bufdt,"R%d.data",cell_start);
      ifstream indata1 (bufdt);              // unwrapped
      indata1>>trash>>orig_com[0]>>orig_com[1]>>orig_com[2];
      for(int i=0;i<nlocal;i++)   indata1>>trash>>orig_x[i][0]>>orig_x[i][1]>>orig_x[i][2];
      indata1.close();

      /* read cell center configurations */
      for(int j=0;j<NN;j++) {
        char buf[128];
        sprintf(buf,"R%d.data",j);
        ifstream indata (buf);
        indata>>trash>>trash>>trash>>trash;
        for(int i=0;i<nlocal;i++)   indata>>trash>>center_x[3*i+0][j]>>center_x[3*i+1][j]>>center_x[3*i+2][j];
        indata.close();
      }
      //initiallize X_0 and V_0 (dw)
      for(int i=0;i<nlocal;i++)   for(int k=0;k<3;k++)  { X_0[i][k] = x[i][k]; V_0[i][k] = v[i][k]; }


    }   // end of initialization activities for first call to fix_nve


//    double t_start = omp_get_wtime();
    // initialize a new trajectory and record findings from finished trajectory
    if(trajectory_finished==1) {  // if the last call to fix_nve concluded a sucessful trajectory from 0 to i+1 
      Tattempt_counter++;
      
      if ((success_index==0 || crossed_interface==0) && unique_trajectories > 0) {
        cout<<"\n----------------------------------------------------\n----------------------------------------------------\n"<<endl; 
        // cout<<" --- recording from cell: "<<picked_m<<endl;
        // for(int i=0;i<nlocal;i++)   for(int k=0;k<3;k++)   { X_0[i][k] = temp_X_0[i][k]; V_0[i][k] = temp_V_0[i][k]; }
         if (old_success_index == 1) ensemble_trajectories++;
         else if (old_success_index == 2) {ensemble_trajectories++; success_counter++;}
         TIS_prob = (double)success_counter/(double)ensemble_trajectories;
         cout<<endl<<" ADDED OLD TRAJECTORY AGAIN"<<endl;
         cout<<" Probability = "<<TIS_prob<<endl;
         cout<<" number of TPE trajectories = "<<ensemble_trajectories<<endl;
         cout<<" successful trajectories "<<success_counter<<endl;
         cout<<" number of unique trajectories = "<<unique_trajectories<<endl;
         cout<<" number of attempts = "<<Tattempt_counter<<endl;
         success_rate = (double)unique_trajectories/(double)Tattempt_counter;
         cout<<" success rate = "<<success_rate<<endl<<endl;
         
        //std::uniform_int_distribution<int> distribution(0,count_hist)
        int pick_seed = rand()%1000000+1;
        ranseed_pointer = &pick_seed;
        int rand_pick = i4_uniform_ab ( 0, count_hist_0, ranseed_pointer );
        cout<<"rand pick = "<<rand_pick<<endl;
        
        for(int i=0;i<nlocal;i++) { 
           for(int k=0;k<3;k++) {
               X_0[i][k] = x_hist_0[i][k][rand_pick];
               V_0[i][k] = v_hist_0[i][k][rand_pick];
          }}
      }  

      //if last trajectory was in TPE then record some things
      if (crossed_interface==1 && success_index>0) {   // successful through Step 5
         double t_end = omp_get_wtime();
         int_time = t_end-t_start;
         cout<<"Time for 1st traj: "<<int_time<<endl;
         unique_trajectories++;
         cout<<"\n----------------------------------------------------\n----------------------------------------------------\n"<<endl;  
         
/*         if (unique_trajectories!=1){
            free(x_hist_0);
            free(v_hist_0);
         }

      x_hist_0 = calloc_2d(nlocal,3, count_hist);
      v_hist_0 = calloc_2d(nlocal,3, count_hist);*/

        for(int i=0;i<nlocal;i++) {
           for(int k=0;k<3;k++) {
              for(int j = 0; j<=count_hist;j++){
                 x_hist_0[i][k][j]=x_hist[i][k][j];v_hist_0[i][k][j]=v_hist[i][k][j];
        }
            }
              }

       
        old_success_index = success_index;

        //if (unique_trajectories==0) flag_1st_traj = 1;

        //cout<<" --- recording from cell: "<<picked_m<<endl;
        //for(int i=0;i<nlocal;i++)   for(int k=0;k<3;k++)   { X_0[i][k] = temp_X_0[i][k]; V_0[i][k] = temp_V_0[i][k]; }

        if (success_index==2) {
          success_counter++;
          //previous_success=1;
        }//else previous_success=0;

        tao_b = step_bck-rsteps;
        tao_f = step_fwd-rsteps;
        time_L_old = tao_b + tao_f;

        // ----- TIS Step 6&7 ------ // redundant
//        time_L_new = tao_b + tao_f; 
//        prob = MIN(1.0,static_cast<double>(time_L_old)/static_cast<double>(time_L_new)); 
//        rand_U = (double)rand()/(double)RAND_MAX;
//        if (rand_U<prob) {
//          for(int i=0;i<nlocal;i++)   for(int k=0;k<3;k++)   { X_0[i][k] = temp_X_0[i][k]; V_0[i][k] = temp_V_0[i][k]; }
//          time_L_old = time_L_new;
//        }
        

        ensemble_trajectories++;
        TIS_prob = (double)success_counter/(double)ensemble_trajectories;
        cout<<" Probability = "<<TIS_prob<<endl;
        cout<<endl<<" number of TPE trajectories = "<<ensemble_trajectories<<endl; 
        cout<<" successful trajectories = "<<success_counter<<endl;
        cout<<" number of unique trajectories = "<<unique_trajectories<<endl;
        cout<<" number of attempts = "<<Tattempt_counter<<endl;
        success_rate = (double)unique_trajectories/(double)Tattempt_counter;
        cout<<" success rate = "<<success_rate<<endl;
        cout<<" number of steps in last trajectory "<<time_L_old<<endl<<endl;
        
        //std::uniform_int_distribution<int> distribution(0,count_hist);
        int pick_seed = rand()%1000000+1;
        ranseed_pointer = &pick_seed;
        int rand_pick = i4_uniform_ab ( 0, count_hist, ranseed_pointer );
        cout<<"count_hist = "<<count_hist<<endl;
        cout<<"rand pick = "<<rand_pick<<endl;
       
        for(int i=0;i<nlocal;i++) {
           for(int k=0;k<3;k++) {
               X_0[i][k] = x_hist_0[i][k][rand_pick];
               V_0[i][k] = v_hist_0[i][k][rand_pick];
          }}

        count_hist_0 = count_hist;
      }//else if(previous_success==1)  { success_counter++; }
        
      count_hist = -1;

 
      //reset flags for new trajectory
      b_index = 0;
      f_index = 0;
      //XV0_rec = 0;
      trajectory_finished = 0;
      crossed_interface = 0;
      success_index = 0;
//pick_index = 0;

      // select cell from which to record starting configuration for the trajectory after this one
//      pnum = 0;
//      do { pnum = (int)rand()%3 - 1; } while (pnum==0);
//      cell_pick = cell_s2p + pnum;
//      if(cell_pick<=2) cell_pick=3;
//      if(cell_pick>=Interface) cell_pick=Interface-1;
//      cell_s2p = cell_pick;

      // set maximum number of steps for this trajectory 
      double alpha = (double)rand()/(double)RAND_MAX;
      N_max = floor(static_cast<double>(time_L_old)/alpha);
      cout<<"N_max = "<<N_max<<endl;

    /*  if (Tattempt_counter != 0){
         free(x_hist);
         free(v_hist);
      }
      x_hist = calloc_2d(nlocal,3, N_max);
      v_hist = calloc_2d(nlocal,3, N_max);*/


      // ----- TIS Step 1 ------ //
      //assign x and v to start new trajectory from
      for(int i=0;i<nlocal;i++)   for(int k=0;k<3;k++)  { x[i][k] = X_0[i][k]; v[i][k] = V_0[i][k]; }

      if (g_step == 0){
         char **arg = new char*[3];
         arg[0] = (char *) "velocity_temp";
         arg[1] = group->names[igroup];
         arg[2] = (char *) "temp";
         //cout<<"arg "<<arg[1]<<endl;

         temper = new ComputeTemp(lmp,3,arg);
         temper->init();
         delete [] arg;}
         double t = temper->compute_scalar();


      cout<<"temperature = "<<t<<endl;



    }    // ------- end of initialization of new trajectory



/*    // XV_0 recording (save positions and velocitites from current trajectory)
    if (XV0_rec==-1) { //from a backward trajectory
      for(int i=0;i<nlocal;i++)   for(int k=0;k<3;k++)   { temp_X_0[i][k] =  x[i][k]; temp_V_0[i][k] = -v[i][k]; }
      XV0_rec = 0;
      picked_m = oldm;
      //cout<<" recording from backward with cell: "<<oldm<<endl;
    }
    if (XV0_rec==1) { // from a forward trajectory
      for(int i=0;i<nlocal;i++)   for(int k=0;k<3;k++)   { temp_X_0[i][k] =  x[i][k]; temp_V_0[i][k] = v[i][k];  }
      XV0_rec = 0;
      picked_m = oldm;
      //cout<<" recording from forward with cell: "<<oldm<<endl;
    }*/

    

    // ------- everything in this if statement can be moved into the bottom of the trajectory finished loop
    if (b_index==0 && f_index==0) {


    // ----- TIS Step 2  ------ //
    //alter momentum

      //measure current ke
      old_ke=0.0;
      for(int i=0;i<nlocal;i++)   for(int k=0;k<3;k++)   { old_ke += v[i][k]*v[i][k]*atom_mass[i]; }
      old_ke*= 0.5*1.0364269e-4; // precesion the same as lammps (dw)

      //measure average velocity mag per atom
      //cout<<"2nd temp = "<<temp<<endl;
      temp=0;
      for(int i=0;i<nlocal;i++)   for(int k=0;k<3;k++)  temp += fabs(v[i][k]);
      velo_magn = temp / static_cast<double>(3*nlocal);
      //cout<<"Velocity magnitude = "<<velo_magn<<endl;
      // record initial H for old time's sake
      if (g_step==0)   H = PE_0 + old_ke;
      
      rand_ensemble = (double)rand()/(double)RAND_MAX;
      cout<<"rand_ensemble = "<<rand_ensemble<<endl;
//      if((rand_ensemble<0.5 && unique_trajectories>0) || (Tattempt_counter>=0 && unique_trajectories==0)){
      if((rand_ensemble<0.5) || unique_trajectories==0){
        rescale=1;
      }else{
        rescale=0;
      }
      cout<<"rescale = "<<rescale<<endl;
      //cout<<"st3_pass = "<<st3_pass<<endl;     
      // add momentum pertabation
      for(int i=0;i<nlocal;i++) {
        for(int k=0;k<3;k++) {
          old_v[i][k] = v[i][k];
          rand_U1 = (double)rand()/(double)RAND_MAX;
          rand_U2 = (double)rand()/(double)RAND_MAX;
          //cout<<"rand_U1 = "<<rand_U1<<endl;
          //cout<<"rand_U2 = "<<rand_U2<<endl;
          if (rand_U1>1e-30)   rand_G=sqrt(-2*log(rand_U1))*cos(2*3.14159*rand_U2);
          else  rand_G=0.0;
          //cout<<"rand_G = "<<rand_G<<endl;
         /* if (Tattempt_counter == 0) {
              trial_v[i][k] = v[i][k];
          }
          else trial_v[i][k] = v[i][k] + 0.0035*2.0*velo_magn*rand_G;*/
          if (rescale == 0) v[i][k] = v[i][k] + 0.005*2.0*velo_magn*rand_G;

          else if(rescale) v[i][k] = v[i][k] + 20*2.0*velo_magn*rand_G;

        }
      }
      zero_momentum(); zero_rotation();
      

//      for(int i=0;i<nlocal;i++) {
  //      for(int k=0;k<3;k++) {trial_v[i][k] = v[i][k];v[i][k] = V_0[i][k];}};
                               

      // measure KE with momentum perturbation
      new_ke=0.0;
      for(int i=0;i<nlocal;i++)   for(int k=0;k<3;k++)   { new_ke += v[i][k]*v[i][k]*atom_mass[i]; }
      new_ke*=0.5*1.0364269e-4; //make precesion the same as lammps (dw)
      ke_added = new_ke - old_ke;
//cout<<"ke_added "<<ke_added<<endl;

      
    // ----- TIS Step 3  ------ //
    // decide whether to accept perturbation 
      prob = MIN(1.0,exp(-ke_added*38.6847));
      rand_U = (double)rand()/(double)RAND_MAX;
      
      //if NVE rescale energy and accept perturbation
      if(rescale){
        for(int i=0;i<nlocal;i++)   for(int k=0;k<3;k++)   v[i][k]*=sqrt(old_ke/new_ke);
        prob=1;
        ke_added=0;
      }
     
      double t = temper->compute_scalar();
      cout<<"temperature = "<<t<<endl;
      
      /*for(int i=0;i<nlocal;i++) {
        for(int k=0;k<3;k++) {trial_v[i][k] = v[i][k];v[i][k] = V_0[i][k];}};*/

      cout<<"Prob = "<<prob<<endl;
      cout<<"dke = "<<ke_added<<endl;
      //if NVT decide whether to perform this trajectory
      if (rand_U<prob) { 
        rescale_check +=1;
        //record starting x and v for bkwd and fwd dynamics
        for(int i=0;i<nlocal;i++)  for(int k=0;k<3;k++)   { st3_x[i][k] = x[i][k]; st3_v[i][k] = v[i][k]; } 
        b_index = 1;
        crossed_interface=0;
        //st3_pass++;
      } 
      else { 
        trajectory_finished=1; 
        cout<<" trajectory ended due to momentum perturbation in NVT "<<endl;
      }

      step_bck = 0;
      step_fwd = 0;

    }
    // end of steps 2 and 3


    // --- Step 4: backward MD
    if (b_index == 1) {

      //hold trajectory in place to ensure neighbor list can catch up to sudden movement
      if (step_bck<rsteps) {
        for(int i=0;i<nlocal;i++)   for(int k=0;k<3;k++)  { x[i][k] = st3_x[i][k]; v[i][k] = -st3_v[i][k]; }
      } else {

        //set velocity on first dynamics step
        if (step_bck==rsteps) {
          for(int i=0;i<nlocal;i++)   for(int k=0;k<3;k++)   v[i][k] = -st3_v[i][k];
          //cout<<endl<<"backward starts - g_step:  "<<g_step<<endl;
        }


        //double t_start = omp_get_wtime();
        //#pragma omp parallel for private(temp)
        // periodicity in x
        for (int i=0;i<nlocal;i++) {
          temp=x[i][0]-orig_x[i][0];
          if (temp>BL[0]*0.8)  { temp -= BL[0]; }
          else if(-temp>BL[0]*0.8) { temp += BL[0]; }
          uw_x[i][0] = temp + orig_x[i][0];
        }
        // free in y
        //#pragma omp for
        for (int i=0;i<nlocal;i++)   uw_x[i][1] = x[i][1];

       // #pragma omp parallel for private(temp)
        // periodicity in z
        for (int i=0;i<nlocal;i++) {
          temp=x[i][2]-orig_x[i][2];
          if (temp>BL[2]*0.8)  { temp -= BL[2]; }
          else if(-temp>BL[2]*0.8) { temp += BL[2]; }
          uw_x[i][2] = temp + orig_x[i][2];
        }
        //#pragma omp parallel for
        for(int i=0;i<nlocal;i++)   for(int k=0;k<3;k++)   old_x[i][k] = uw_x[i][k];
        //}
        //double t_end = omp_get_wtime();
        //int_time = int_time + (t_end-t_start);
        //if (g_step == 3000)
        //cout<<"The time is: "<<int_time<<endl;

        if (rmass) {
         // #pragma omp for private(dtfm)
          for (int i = 0; i < nlocal; i++) {
            if (mask[i] & groupbit) {
  	      dtfm = dtf / rmass[i];
	      v[i][0] += dtfm * f[i][0];
	      v[i][1] += dtfm * f[i][1];
	      v[i][2] += dtfm * f[i][2];
	      uw_x[i][0] += dtv * v[i][0];
	      uw_x[i][1] += dtv * v[i][1];
	      uw_x[i][2] += dtv * v[i][2];
            }
          }
        } else {
          //#pragma omp for private(dtfm)
          for (int i = 0; i < nlocal; i++) {
            if (mask[i] & groupbit) {
	      dtfm = dtf / mass[type[i]];
	      v[i][0] += dtfm * f[i][0];
	      v[i][1] += dtfm * f[i][1];
	      v[i][2] += dtfm * f[i][2];
	      uw_x[i][0] += dtv * v[i][0];
	      uw_x[i][1] += dtv * v[i][1];
	      uw_x[i][2] += dtv * v[i][2];
            }
          }
        }
        for(int i=0;i<nlocal;i++) for(int k=0;k<3;k++)  x[i][k] += (uw_x[i][k]-old_x[i][k]);

        //double t_start = omp_get_wtime();
        // measure distance between current configuration and cell centers
        for(int j=0;j<NN;j++) { 
          vor_dist[j]=0.0;
          for(int i=0;i<nlocal;i++)  for(int k=0;k<3;k++)   { temp = uw_x[i][k] - center_x[3*i+k][j]; vor_dist[j] += temp*temp; }
          vor_dist[j]=sqrt(vor_dist[j]);
        }
        
        /*double t_end = omp_get_wtime();
        int_time = int_time + (t_end-t_start);
        if (g_step == 3000)
        cout<<"Time is: "<<int_time<<endl;*/

        // determine which cell the current configuration is in
        vor_check = vor_dist[0];
        for(int m=0;m<NN;m++)   vor_check = MIN(vor_check,vor_dist[m]);

/*
for(int m=0;m<NN;m++)  { 
  if (fabs(vor_dist[m]-vor_check)<1.0e-10 && m!=oldm){   
    cout<<m<<" -- back_step:"<<step_bck<<" -- g_step:"<<g_step<<endl; 
    oldm=m;
  }
}
*/

        // if in ground cell then stop bkwd MD and start fwd MD
        //if (fabs(vor_dist[0]-vor_check)<1.0e-10 && oldm==1)   {
        if ( (fabs(vor_dist[0]-vor_check)<1.0e-10 || fabs(vor_dist[1]-vor_check)<1.0e-10 || fabs(vor_dist[2]-vor_check)<1.0e-10 || fabs(vor_dist[3]-vor_check)<1.0e-10) && oldm==4 )   {
           b_index = 0; f_index = 1;
           cout<<"backwards MD suceeded "<<g_step<<" "<<step_bck<<endl;
        }
        else if (fabs(vor_dist[lamda+2]-vor_check)<1.0e-10 && oldm==lamda+1 && success_counter>0)   {  //assumed first trajectory will be a success
        //else if ( (fabs(vor_dist[lamda+2]-vor_check)<1.0e-10 || fabs(vor_dist[lamda+3]-vor_check)<1.0e-10 || fabs(vor_dist[lamda+4]-vor_check)<1.0e-10) && oldm==lamda+1 && success_counter>0 )   {
          trajectory_finished=1; 
          cout<<"backwards MD failed, back to step 1 "<<g_step<<" "<<step_bck<<endl;
        }

        // check if sim makes it into next to last cell (meaning it crossed the interface lam_i
        if (crossed_interface==0 && fabs(vor_dist[lamda]-vor_check)<1.0e-10 && oldm==lamda+1)   { crossed_interface = 1; }
        if (crossed_interface==0 && fabs(vor_dist[lamda+1]-vor_check)<1.0e-10 && oldm==lamda)   { crossed_interface = 1; }

        // record the number of the cell containing the current position
        for(int m=0;m<NN;m++)   if(fabs(vor_dist[m]-vor_check)<1.0e-10)   oldm=m;

        if ((step_bck-rsteps)>=0 && (step_bck-rsteps)%10 == 0 && count_hist <N_hist){
           count_hist += 1;
           for(int i=0;i<nlocal;i++) {for(int k=0;k<3;k++)  {x_hist[i][k][count_hist]=x[i][k];v_hist[i][k][count_hist]=-v[i][k];} }}
        
        /* pick XV_0 for next TIS step */
        //if (fabs(vor_dist[cell_pick]-vor_check)<1.0e-10 && pick_index==0) {
        //  XV0_rec = -1;
        //  pick_index = 1;
        //}
        /*prob = 1.0/static_cast<double>(step_bck-rsteps+1);
        rand_U = (double)rand()/(double)RAND_MAX;
        if (rand_U<prob)   XV0_rec = -1;*/

     } 

      // check if sim has gone too many steps..if so then stop this trajectory 
      if (static_cast<double>(step_bck)-rsteps > N_max) { 
        trajectory_finished=1;
        cout<<"backwards MD failed due to N_max, back to step 3 "<<step_bck-rsteps<<"  N_max="<<N_max<<endl;
      }
     
      step_bck++;
      
    }



    // --- Step 5: forward MD
    if (f_index == 1) {
      if (step_fwd<rsteps) {
        for(int i=0;i<nlocal;i++)  for(int k=0;k<3;k++)  { x[i][k] = st3_x[i][k]; v[i][k] = st3_v[i][k]; }
      } else {

        if (step_fwd==rsteps) {
          for(int i=0;i<nlocal;i++)  for(int k=0;k<3;k++)  v[i][k] = st3_v[i][k];
          //cout<<endl<<"forward starts - g_step:  "<<g_step<<endl;
        }

        // periodicity in x
        for (int i=0;i<nlocal;i++) {
          temp=x[i][0]-orig_x[i][0];
          if (temp>BL[0]*0.8)  { temp -= BL[0]; }
          else if(-temp>BL[0]*0.8) { temp += BL[0]; }
          uw_x[i][0] = temp + orig_x[i][0];
        }
        // free in y
        for (int i=0;i<nlocal;i++)   uw_x[i][1] = x[i][1];
        // periodicity in z
        for (int i=0;i<nlocal;i++) {
          temp=x[i][2]-orig_x[i][2];
          if (temp>BL[2]*0.8)  { temp -= BL[2]; }
          else if(-temp>BL[2]*0.8) { temp += BL[2]; }
          uw_x[i][2] = temp + orig_x[i][2];
        }
        for(int i=0;i<nlocal;i++)   for(int k=0;k<3;k++)   old_x[i][k] = uw_x[i][k];

        if (rmass) {
          for (int i = 0; i < nlocal; i++) {
            if (mask[i] & groupbit) {
              dtfm = dtf / rmass[i];
              v[i][0] += dtfm * f[i][0];
              v[i][1] += dtfm * f[i][1];
              v[i][2] += dtfm * f[i][2];
              uw_x[i][0] += dtv * v[i][0];
              uw_x[i][1] += dtv * v[i][1];
              uw_x[i][2] += dtv * v[i][2];
            }
          }
        } else {
          for (int i = 0; i < nlocal; i++) {
            if (mask[i] & groupbit) {
              dtfm = dtf / mass[type[i]];
              v[i][0] += dtfm * f[i][0];
              v[i][1] += dtfm * f[i][1];
              v[i][2] += dtfm * f[i][2];
              uw_x[i][0] += dtv * v[i][0];
              uw_x[i][1] += dtv * v[i][1];
              uw_x[i][2] += dtv * v[i][2];
            }
          }
        }
        for(int i=0;i<nlocal;i++) for(int k=0;k<3;k++)  x[i][k] += (uw_x[i][k]-old_x[i][k]);
     

        //measure distance to cell centers from current configuration 
        for(int j=0;j<NN;j++) {
          vor_dist[j]=0.0;
          for(int i=0;i<nlocal;i++)  for(int k=0;k<3;k++)   { temp = uw_x[i][k] - center_x[3*i+k][j]; vor_dist[j] += temp*temp; }
          vor_dist[j]=sqrt(vor_dist[j]);
        }

        //determine which cell the configuration is in
        vor_check = vor_dist[0];
        for(int m=0;m<NN;m++)   vor_check = MIN(vor_check,vor_dist[m]);

/*
for(int m=0;m<NN;m++)  { 
  if (fabs(vor_dist[m]-vor_check)<1.0e-10 && m!=oldm){   
    cout<<m<<" -- fwd_step:"<<step_bck<<" -- g_step:"<<g_step<<endl; 
    oldm=m;
  }
}
*/

        // dww added step6_index=1 in else if
        if (fabs(vor_dist[lamda+2]-vor_check)<1.0e-10 && oldm==lamda+1)   {
        //if ( (fabs(vor_dist[lamda+2]-vor_check)<1.0e-10 || fabs(vor_dist[lamda+3]-vor_check)<1.0e-10 || fabs(vor_dist[lamda+4]-vor_check)<1.0e-10) && oldm==lamda+1 )   {
          trajectory_finished=1; 
          success_index = 2; 
          cout<<"forward MD made it to lam_i+1 "<<g_step<<" "<<step_fwd<<endl;
        //} else if(fabs(vor_dist[0]-vor_check)<1.0e-10 && oldm==1)   {
        } else if( (fabs(vor_dist[0]-vor_check)<1.0e-10 || fabs(vor_dist[1]-vor_check)<1.0e-10 || fabs(vor_dist[2]-vor_check)<1.0e-10  || fabs(vor_dist[3]-vor_check)<1.0e-10) && oldm==4 )   {
          trajectory_finished=1;
          success_index = 1;
          cout<<"forward MD made it to lam_0  "<<g_step<<" "<<step_fwd<<endl;
        }

        // dww check if sim makes it into next to last cell (meaning it crossed the interface lam_i
        if (crossed_interface==0 && fabs(vor_dist[lamda]-vor_check)<1.0e-10 && oldm==lamda+1)   { crossed_interface = 1; }
        if (crossed_interface==0 && fabs(vor_dist[lamda+1]-vor_check)<1.0e-10 && oldm==lamda)   { crossed_interface = 1; }

        // record the number of the cell containing the current position
        for(int m=0;m<NN;m++)   if(fabs(vor_dist[m]-vor_check)<1.0e-10)   oldm=m;

        /* pick XV_0 for next TIS step */
        //if (fabs(vor_dist[cell_pick]-vor_check)<1.0e-10 && pick_index==0) {
        //  XV0_rec = 1;
        //  pick_index = 1;
        //}
        /*prob = 1.0/static_cast<double>(step_fwd+step_bck-2*rsteps+2);
        rand_U = (double)rand()/(double)RAND_MAX;
        if (rand_U<prob)   XV0_rec = 1;*/
}
      
      
      if ((step_fwd-rsteps)>=0 && (step_fwd-rsteps)%10 == 0 && count_hist <=2*N_hist){
           count_hist += 1;
           for(int i=0;i<nlocal;i++) {for(int k=0;k<3;k++)  {x_hist[i][k][count_hist]=x[i][k];v_hist[i][k][count_hist]=v[i][k];} }}
      //added the pick_index (dw)
      if (static_cast<double>(step_bck+step_fwd-2*rsteps) > N_max) {
        trajectory_finished=1; 
        cout<<"forward MD failed due to N_max, back to step 1 "<<step_bck+step_fwd-2*rsteps<<" N_max="<<N_max<<endl;
      }

      step_fwd++;

    }
    // end of fwd MD


/*        double t_end = omp_get_wtime();
        int_time = int_time + (t_end-t_start);
        cout<<"Time is: "<<t_end-t_start<<endl;*/

} //------ final end

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
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

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

  g_step++;

}

/* ---------------------------------------------------------------------- */

void FixNVE::initial_integrate_respa(int vflag, int ilevel, int flag)
{
  if (flag) return;             // only used by NPT,NPH

  dtv = step_respa[ilevel];
  dtf = 0.5 * step_respa[ilevel] * force->ftm2v;

  // innermost level - NVE update of v and x
  // all other levels - NVE update of v

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
}

void FixNVE::zero_momentum()
{
  // cannot have 0 atoms in group

  if (group->count(igroup) == 0.0)
    error->all("Cannot zero momentum of 0 atoms");

  // compute velocity of center-of-mass of group

  double masstotal = group->mass(igroup);
  double vcm[3];
  group->vcm(igroup,masstotal,vcm);

  // adjust velocities by vcm to zero linear momentum

  double **v = atom->v;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      v[i][0] -= vcm[0];
      v[i][1] -= vcm[1];
      v[i][2] -= vcm[2];
    }
}
void FixNVE::zero_rotation()
{
  int i;

  // cannot have 0 atoms in group

  if (group->count(igroup) == 0.0)
    error->all("Cannot zero momentum of 0 atoms");

  // compute omega (angular velocity) of group around center-of-mass

  double xcm[3],angmom[3],inertia[3][3],omega[3];
  double masstotal = group->mass(igroup);
  group->xcm(igroup,masstotal,xcm);
  group->angmom(igroup,xcm,angmom);
  group->inertia(igroup,xcm,inertia);
  group->omega(angmom,inertia,omega);

  // adjust velocities to zero omega
  // vnew_i = v_i - w x r_i
  // must use unwrapped coords to compute r_i correctly

  double **x = atom->x;
  double **v = atom->v;
  int *mask = atom->mask;
  int *image = atom->image;
  int nlocal = atom->nlocal;

  int xbox,ybox,zbox;
  double dx,dy,dz;
  double xprd = domain->xprd;
  double yprd = domain->yprd;
  double zprd = domain->zprd;

  for (i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      xbox = (image[i] & 1023) - 512;
      ybox = (image[i] >> 10 & 1023) - 512;
      zbox = (image[i] >> 20) - 512;
      dx = (x[i][0] + xbox*xprd) - xcm[0];
      dy = (x[i][1] + ybox*yprd) - xcm[1];
      dz = (x[i][2] + zbox*zprd) - xcm[2];
      v[i][0] -= omega[1]*dz - omega[2]*dy;
      v[i][1] -= omega[2]*dx - omega[0]*dz;
      v[i][2] -= omega[0]*dy - omega[1]*dx;
    }
}
void FixNVE::rescale_vel_temp(double t_old, double t_new)
{
  if (t_old == 0.0) error->all("Attempting to rescale a 0.0 temperature");

  double factor = sqrt(t_new/t_old);

  double **v = atom->v;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      v[i][0] *= factor;
      v[i][1] *= factor;
      v[i][2] *= factor;
    }
}

int FixNVE::i4_uniform_ab ( int a, int b, int *seed )

/******************************************************************************/
/*
  Purpose:

    I4_UNIFORM_AB returns a scaled pseudorandom I4 between A and B.

  Discussion:

    The pseudorandom number should be uniformly distributed
    between A and B.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    24 May 2012

  Author:

    John Burkardt

  Reference:

    Paul Bratley, Bennett Fox, Linus Schrage,
    A Guide to Simulation,
    Second Edition,
    Springer, 1987,
    ISBN: 0387964673,
    LC: QA76.9.C65.B73.

    Bennett Fox,
    Algorithm 647:
    Implementation and Relative Efficiency of Quasirandom
    Sequence Generators,
    ACM Transactions on Mathematical Software,
    Volume 12, Number 4, December 1986, pages 362-376.

    Pierre L'Ecuyer,
    Random Number Generation,
    in Handbook of Simulation,
    edited by Jerry Banks,
    Wiley, 1998,
    ISBN: 0471134031,
    LC: T57.62.H37.

    Peter Lewis, Allen Goodman, James Miller,
    A Pseudo-Random Number Generator for the System/360,
    IBM Systems Journal,
    Volume 8, Number 2, 1969, pages 136-143.

  Parameters:

    Input, int A, B, the limits of the interval.

    Input/output, int *SEED, the "seed" value, which should NOT be 0.
    On output, SEED has been updated.

    Output, int I4_UNIFORM_AB, a number between A and B.
*/
{
  int c;
  int i4_huge = 2147483647;
  int k;
  float r;
  int value;

  if ( *seed == 0 )
  {
    fprintf ( stderr, "\n" );
    fprintf ( stderr, "I4_UNIFORM_AB - Fatal error!\n" );
    fprintf ( stderr, "  Input value of SEED = 0.\n" );
    exit ( 1 );
  }
/*
  Guaranteee A <= B.
*/
  if ( b < a )
  {
    c = a;
    a = b;
    b = c;
  }

  k = *seed / 127773;

  *seed = 16807 * ( *seed - k * 127773 ) - k * 2836;

  if ( *seed < 0 )
  {
    *seed = *seed + i4_huge;
  }

  r = ( float ) ( *seed ) * 4.656612875E-10;
/*
  Scale R to lie between A-0.5 and B+0.5.
*/
  r = ( 1.0 - r ) * ( ( float ) ( a ) - 0.5 ) 
    +         r   * ( ( float ) ( b ) + 0.5 );
/*
  Round R to the nearest integer.
*/
  value = round ( r );
/*
  Guarantee that A <= VALUE <= B.
*/
  if ( value < a )
  {
    value = a;
  }
  if ( b < value )
  {
    value = b;
  }

  return value;
}

double*** FixNVE::calloc_2d(long int l,long int m, long int n)	//allocate a double matrix
{
double*** array;    // 3D array definition;
// begin memory allocation
array = new double**[l];
for(long int x = 0; x < l; ++x) {
    array[x] = new double*[m];
    for(long int y = 0; y < m; ++y) {
        array[x][y] = new double[n];
        for(long int z = 0; z < n; ++z) { // initialize the values, again, not necessary, but recommended
            array[x][y][z] = 0;
        }
    }
}
return array;

}
