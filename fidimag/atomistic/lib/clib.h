#ifndef __CLIB__
#define __CLIB__

#include<math.h>
#include<complex.h>
#include<fftw3.h>
//#include<omp.h>

#define WIDE_PI 3.1415926535897932384626433832795L

double single_random(void);
void gauss_random_vec(double *x, int n);
void random_spin_uniform(double *spin, int n);

/* 3 components for the cross product calculations */
inline double cross_x(double a0, double a1, double a2,
                      double b0, double b1, double b2) { return a1 * b2 - a2 * b1; }
inline double cross_y(double a0, double a1, double a2,
                      double b0, double b1, double b2) { return a2 * b0 - a0 * b2; }
inline double cross_z(double a0, double a1, double a2,
                      double b0, double b1, double b2) { return a0 * b1 - a1 * b0; }

void compute_exch_field(double *spin, double *field, double *energy,
						double Jx, double Jy, double Jz,
                        int *ngbs, int n);


void compute_exch_field_spatial(double *spin, double *field, double *energy,double *J,
                        int *ngbs, int n);

double compute_exch_energy(double *spin, double Jx, double Jy, double Jz,
                           int nx, int ny, int nz,
                           int xperiodic, int yperiodic);

void compute_anis(double *spin, double *field, double *energy,
	              double *Ku, double *axis, int n);

void dmi_field_bulk(double *spin, double *field, double *energy,
                    double *D, int *ngbs, int n);

void dmi_field_interfacial_atomistic(double *spin, double *field,
                                     double *energy, double D, int *ngbs,
                                     int n, int nneighbours,
                                     double *DMI_vec);

void demag_full(double *spin, double *field, double *energy,
                double *coords,
                double *mu_s, double *mu_s_scale, int n);

double dmi_energy(double *spin, double D, int nx, int ny, int nz,
                  int xperiodic, int yperiodic);

void llg_rhs(double * dm_dt, double * spin, double * h, double *alpha,
		     int *pins, double gamma, int n, int do_precession,
             double default_c);

void llg_rhs_jtimes(double *jtn, double *m, double *h,
                    double *mp, double *hp, double *alpha, int *pins,
                    double gamma, int n, int do_precession, double default_c);

void llg_s_rhs(double * dm_dt, double * spin, double * h, double *alpha,
             double *chi, double gamma, int n);


void compute_stt_field_c(double *spin, double *field, double *jx, double *jy,
		double dx, double dy, int *ngbs, int n);

void llg_stt_rhs(double *dm_dt, double *m, double *h,
                 double *h_stt, double *alpha,
                 double beta, double u0, double gamma, int n);

void llg_stt_cpp(double *dm_dt, double *m, double *h, double *p,
			      double *alpha, int *pins, double *a_J, double beta, double gamma, int n);


void normalise(double *m, int *pins, int n);

double skyrmion_number(double *spin, double *charge,
                       int nx, int ny, int nz, int *ngbs);

void compute_guiding_center(double *spin, int nx, int ny, int nz, double *res);

void compute_px_py_c(double *spin, int nx, int ny, int nz,
                     double *px, double *py);

//======================================================================

void llg_rhs_dw_c(double *m, double *h, double *dm, double *T, double *alpha, 
                  double *mu_s_inv, int *pins, double *eta, int n, double gamma, double dt);

//======================================================================
void run_step_mc(double *spin, double *new_spin, int *ngbs, double J, double D, double *h, int n, double T);



#endif
