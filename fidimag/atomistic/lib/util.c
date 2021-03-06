#include "clib.h"

//compute the S \cdot (S_i \times S_j)
inline double volume(double S[3], double Si[3], double Sj[3]) {
	double tx = S[0] * (-Si[2] * Sj[1] + Si[1] * Sj[2]);
	double ty = S[1] * (Si[2] * Sj[0] - Si[0] * Sj[2]);
	double tz = S[2] * (-Si[1] * Sj[0] + Si[0] * Sj[1]);
	return tx + ty + tz;
}

double skyrmion_number(double *spin, double *charge,
                       int nx, int ny, int nz, int *ngbs) {

    /* Calculation of the "Skyrmion number" Q for a two dimensional discrete
     * spin lattice in the x-y plane (also known
     * as finite spin chirality)
     *
     * The *spin array is the vector field for a two dimensional
     * lattice with dimensions nx * ny
     * (we can take a slice of a bulk from Python and pass it here, 
     *  remember to do the ame for the neighbours matrix)
     * The array follows the order:
     *   [Sx0 Sy0 Sz0 Sx1 Sy1 Sz1 ... ]
     *
     * Charge is a scalar field array used to store the spin chirality /
     * skyrmion number density (Q value per lattice site)
     *
     * *ngbs is the array with the neighbours information for every
     * lattice site. The array is like:
     *      [ 0-x, 0+x, 0-y, 0+y, 0-z, 0+z, 1-x, 1+x, 1-y, ...  ]
     *        i=0                           i=1                ...
     *
     * where  0-y  is the index of the neighbour of the 0th spin,
     * in the -y direction, for example
     *
     *
     * THEORY:
     *
     * Referring to the i-th site of a
     * square lattice, we refer to the x direction for the i-th neighbours
     * and to the y direction for the j-th neighbours, as here:
     *
     *                    X  (j + 1)
     *                    |
     *                (i) |
     *     (i - 1) X------O------X   (i + 1)
     *                    |
     *                    |
     *                    X   (j - 1)
     *
     * Then, we use the follwing expression:
     *
     * Q =  S_i \dot ( S_{i+1} \times S_{j+1} ) 
     *      +  S_i \dot ( S_{i-1} \times S_{j-1} )
     *
     * This expression is based on the publication PRL 108, 017601 (2012)
     * where Q is called "finite spin chirality". The idea comes from
     * discrete chiral quantities in Hall effect studies. For example, at
     * the end of page 3 in Rep. Prog. Phys. 78 (2015) 052502, it
     * is argued:
     *     scalar chirality (...) , which measures the volume enclosed 
     *     by the three spins of the elementary triangle and, similarly to
     *     (the vector chirlity) is sensitive to the sense of spin's 
     *     rotation in the x–y plane
     *
     *  Hence we are taking the triangles formed by (i, i+1, j+1)
     *  and (i, i-1, j-1) whose total area covers a unit cell,
     *  and ommit the other two triangles (bottom right and top left)
     *  When we sum this quantity across the whole lattice we cover every
     *  triangle. The final quantity will be scaled by 8 PI
     *  to match +-1 for a full skyrmion configuration
     *
     *  Recently, other ways to calculate a discrete skyrmion number have
     *  been proposed: http://arxiv.org/pdf/1601.08212.pdf
     *                 Phys. Rev. B 93, 024417 
     *  
     *  also based on using three spins using triangles. This could be
     *  useful for applying to a hexagonal lattice in the future. 
     *
     */  

	int i, j;
	int index, id;

	double sum = 0;

	double S[3], S_i[3], S_j[3];

	int nxy = nx * ny;

	for (i = 0; i < nxy; i++) {
        index = 3 * i;

        /* The starting index of the nearest neighbours for the 
         * i-th spin */
        int id_nn = 6 * i;

        S[0] = spin[index];
        S[1] = spin[index + 1];
        S[2] = spin[index + 2];

        S_i[0] = S_i[1] = S_i[2] = 0;
        S_j[0] = S_j[1] = S_j[2] = 0;

        // neighbour at -x
        // Remember that the index is -1 for sites without material
        if (ngbs[id_nn] > 0) {
            id = 3 * ngbs[id_nn];
            S_i[0] = spin[id];
            S_i[1] = spin[id + 1];
            S_i[2] = spin[id + 2];
        }

        // neighbour at -y
        if (ngbs[id_nn + 2] > 0) {
            id = 3 * ngbs[id_nn + 2];
            S_j[0] = spin[id];
            S_j[1] = spin[id + 1];
            S_j[2] = spin[id + 2];
        }

        // The  S_i \dot ( S_{i+1} \times S_{j+1} )
        charge[i] = volume(S, S_i, S_j);

        S_i[0] = S_i[1] = S_i[2] = 0;
        S_j[0] = S_j[1] = S_j[2] = 0;

        // neighbour at +x
        if (ngbs[id_nn + 1] > 0) {
            id = 3 * ngbs[id_nn + 1];
            S_i[0] = spin[id];
            S_i[1] = spin[id + 1];
            S_i[2] = spin[id + 2];
        }

        // neighbour at +y
        if (ngbs[id_nn + 3] > 0) {
            id = 3 * ngbs[id_nn + 3];
            S_j[0] = spin[id];
            S_j[1] = spin[id + 1];
            S_j[2] = spin[id + 2];
        }

        //  The S_i \dot ( S_{i-1} \times S_{j-1} )
        charge[i]  += volume(S, S_i, S_j);

        /* Scale the chirality quantity */
        charge[i] /= (8 * WIDE_PI);

        /* We use the sum to output the total spin chirality
         * or skyrmion number */
        sum += charge[i];
    }

	return sum;

}

//compute the first derivative respect to x and for the whole mesh
//assume 2d pbc is used
void compute_px_py_c(double *spin, int nx, int ny, int nz, double *px, double *py){
	int nyz = ny * nz;
	int n1 = nx * nyz, n2 = 2 * n1;

	for(int i=0;i<nx; i++){
		for (int j = 0; j < ny; j++) {
			int index = nyz * i + nz * j;


			//x-1
			int id1 = index - nyz;
            if (i==0) {
                id1 += n1;
            }

			//x+1
			int id2 = index + nyz;
            if (i == nx-1){
            	id2 -= n1;
            }

			px[index] = (spin[id2]-spin[id1])/2.0;
			px[index+n1] = (spin[id2 + n1]-spin[id1 + n1])/2.0;
			px[index+n2] = (spin[id2 + n2]-spin[id1 + n2])/2.0;

			//y-1
			id1 = index - nz;
            if (j==0) {
                id1 += nyz;
            }

			//y+1
			id2 = index + nz;
            if (j == ny-1){
                id2 -= nyz;
            }

           	py[index] = (spin[id2]-spin[id1])/2.0;
			py[index+n1] = (spin[id2 + n1]-spin[id1 + n1])/2.0;
			py[index+n2] = (spin[id2 + n2]-spin[id1 + n2])/2.0;
		}
	}

}

// compute the guiding centre, Dynamics of magnetic vortices,     N. Papanicolaou,
// T.N. Tomaras 360, 425-462, (1991)
void compute_guiding_center(double *spin, int nx, int ny, int nz, double *res) {

	int nyz = ny * nz;
	int n1 = nx * nyz, n2 = 2 * n1;
	int i, j;
	int index, id;

	double charge;
	double sum = 0, Rx = 0, Ry = 0;

	double S[3], S_i[3], S_j[3];

	for (i = 0; i < nx; i++) {
		for (j = 0; j < ny; j++) {
			index = nyz * i + nz * j;
			S[0] = spin[index];
			S[1] = spin[index + n1];
			S[2] = spin[index + n2];

			S_i[0] = S_i[1] = S_i[2] = 0;
			S_j[0] = S_j[1] = S_j[2] = 0;
			if (j > 0) {
				id = index - nz;
				S_j[0] = spin[id];
				S_j[1] = spin[id + n1];
				S_j[2] = spin[id + n2];
			}

			if (i > 0) {
				id = index - nyz;
				S_i[0] = spin[id];
				S_i[1] = spin[id + n1];
				S_i[2] = spin[id + n2];
			}

			charge = volume(S, S_i, S_j);
			sum += charge;
			Rx += i*charge;
			Ry += j*charge;

			S_i[0] = S_i[1] = S_i[2] = 0;
			S_j[0] = S_j[1] = S_j[2] = 0;
			if (i < nx - 1 ) {
				id = index + nyz;
				S_i[0] = spin[id];
				S_i[1] = spin[id + n1];
				S_i[2] = spin[id + n2];
			}

			if (j < ny - 1) {
				id = index + nz;
				S_j[0] = spin[id];
				S_j[1] = spin[id + n1];
				S_j[2] = spin[id + n2];
			}


			charge = volume(S, S_i, S_j);
			sum += charge;
			Rx += i*charge;
			Ry += j*charge;
		}
	}

	res[0]=Rx/sum;
	res[1]=Ry/sum;

}
