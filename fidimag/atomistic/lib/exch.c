#include "clib.h"

/*
 compute the effective exchange field at site i
 
 H_i = J \sum_<i,j> S_j
 
 with Hamiltonian
 
 Hamiltonian = - J \sum_<i,j> S_i \cdot S_j
 
 Note that the pair <i,j> only run once for each pair.
 
 */
void compute_exch_field(double *spin, double *field, double *energy,
						double Jx, double Jy, double Jz,
                        int *ngbs, int n) {
    
    #pragma omp parallel for
	for (int i = 0; i < n; i++) {

		int id = 0;
		int id_nn = 6 * i; // index for the neighbours
		
		double fx = 0, fy = 0, fz = 0;

        /* ngbs[] contains the indexes of the neighbours
         * in the following order:
         *      -x, +x, -y, +y, -z, +z
         *  
         * for every spin. It is -1 for boundaries.
         * The array is like:
         *      | 0-x, 0+x, 0-y, 0+y, 0-z, 0+z, 1-x, 1+x, 1-y, ...  |
         *        i=0                           i=1                ...
         *
         * where  0-y  is the index of the neighbour of the 0th spin,
         * in the -y direction, for example
         * 
         * Thus, for every neighbour ( ngbs[i + j], j=0,1,...5 )
         * we compute the field contribution: Sum_j J_j S_j
         * of the neighbours to the exchange interaction
         *
         * The ngbs array also gives the correct indexes for the spins
         * at periodic boundaries
         */

        for (int j = 0; j < 6; j++) {
            
            if (ngbs[id_nn + j] >= 0) {

                id = 3 * ngbs[id_nn + j]; 
                fx += Jx * spin[id];
                fy += Jy * spin[id + 1];
                fz += Jz * spin[id + 2];
            }

        }

        field[3 * i] = fx;
        field[3 * i + 1] = fy;
        field[3 * i + 2] = fz;
        energy[i] = -0.5 * (fx * spin[3 * i] + fy * spin[3 * i + 1] + 
                            fz * spin[3 * i + 2]);
    }
}



double compute_exch_energy(double *spin, double Jx,  double Jy, double Jz,
			int nx, int ny, int nz, int xperiodic, int yperiodic) {
    
	int nyz = ny * nz;
	int n1 = nx * nyz, n2 = 2 * n1;
	int i, j, k;
	int index, id;
	double Sx, Sy, Sz;
	double energy = 0;
    
	for (i = 0; i < nx; i++) {
        for (j = 0; j < ny; j++) {
            for (k = 0; k < nz; k++) {
                index = nyz * i + nz * j + k;
                Sx = spin[index];
                Sy = spin[index + n1];
                Sz = spin[index + n2];
                
                if (i < nx - 1 || xperiodic) {
                    id = index + nyz;
                    if (i == nx-1){
                        id -= n1;
                    }
                    energy += Jx * Sx * spin[id];
                    energy += Jy * Sy * spin[id + n1];
                    energy += Jz * Sz * spin[id + n2];
                }
                
                if (j < ny - 1 || yperiodic) {
                    id = index + nz;
                    if (j == ny-1){
                        id -= nyz;
                    }
                    energy += Jx * Sx * spin[id];
                    energy += Jy * Sy * spin[id + n1];
                    energy += Jz * Sz * spin[id + n2];
                }
                
                if (k < nz - 1) {
                    id = index + 1;
                    energy += Jx * Sx * spin[id];
                    energy += Jy * Sy * spin[id + n1];
                    energy += Jz * Sz * spin[id + n2];
                }
                
            }
        }
	}
    
	energy = -energy;
    
	return energy;
    
}


/*
 compute the effective exchange field at site i
 
 H_i =  \sum_<i,j> J_{ij} S_j
 
 with Hamiltonian
 
 Hamiltonian = - \sum_<i,j> J_{ij} S_i \cdot S_j
 
 Note that the pair <i,j> only run once for each pair.
 
 */
void compute_exch_field_spatial(double *spin, double *field, double *energy,
				double *J, int *ngbs, int n) {
    
    #pragma omp parallel for
	for (int i = 0; i < n; i++) {

		int id = 0;
		int id_nn = 6 * i; // index for the neighbours
		
		double fx = 0, fy = 0, fz = 0;

        /* ngbs[] contains the indexes of the neighbours
         * in the following order:
         *      -x, +x, -y, +y, -z, +z
         *  
         * for every spin. It is -1 for boundaries.
         * The array is like:
         *      | 0-x, 0+x, 0-y, 0+y, 0-z, 0+z, 1-x, 1+x, 1-y, ...  |
         *        i=0                           i=1                ...
         *
         * where  0-y  is the index of the neighbour of the 0th spin,
         * in the -y direction, for example
         * 
         * Thus, for every neighbour ( ngbs[i + j], j=0,1,...5 )
         * we compute the field contribution: Sum_j J_j S_j
         * of the neighbours to the exchange interaction
         *
         * The ngbs array also gives the correct indexes for the spins
         * at periodic boundaries
         *
         * The array J has the same size of array the ngbs.
         */

        for (int j = 0; j < 6; j++) {
            
            int p = id_nn + j;
            
            if (ngbs[p] >= 0) {

                id = 3 * ngbs[p]; 
                fx += J[p] * spin[id];
                fy += J[p] * spin[id + 1];
                fz += J[p] * spin[id + 2];
            }

        }

        field[3 * i] = fx;
        field[3 * i + 1] = fy;
        field[3 * i + 2] = fz;
        energy[i] = -0.5 * (fx * spin[3 * i] + fy * spin[3 * i + 1] + 
                            fz * spin[3 * i + 2]);
    }
}
