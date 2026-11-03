// GVDW kernel checks: series / closed-form continuity of G, forces against finite differences, and the
// energy of a box (tests/gvdw_input.py) with pmemd-pgm's global parameters, for comparison with
// pmemd-pgm's VDWAALS.
//   g++ -O2 -std=c++17 -I.. test_gvdw.cpp -o test_gvdw
//   ./test_gvdw [box.txt rc_nm]
#include "gvdw_kernel.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace pgmcore;
using namespace std;

static const double KCAL = 4.184;

int main(int argc, char** argv) {
    int bad = 0;
    // 1. G and H across the switch at y = 0.6
    double G1, H1, G2, H2;
    gvdwG(0.6-1e-12, G1, H1);
    gvdwG(0.6+1e-12, G2, H2);
    printf("G at y=0.6: series %.15g closed %.15g (rel %.2e); H: %.15g %.15g (rel %.2e)\n", G1, G2,
           fabs(G1-G2)/fabs(G2), H1, H2, fabs(H1-H2)/fabs(H2));
    if (fabs(G1-G2) > 1e-9*fabs(G2) || fabs(H1-H2) > 1e-7*fabs(H2))
        bad++;
    // 2. dU/dr against central differences, both repulsion forms, small and large y
    for (int rep = 0; rep < 2; rep++)
        for (double r : {0.02, 0.05, 0.1, 0.2, 0.28, 0.35, 0.5, 0.9}) {
            double beta = 1.0/sqrt(2.0*(0.07*0.07+0.07*0.07)), A = 800.0, C6 = 0.0025, B = (rep ? 4.52 : 0.9453);
            double dEdR, h = 1e-6, dp, dm;
            gvdwPair(r, beta, A, C6, B, rep, dEdR);
            double fd = -(gvdwPair(r+h, beta, A, C6, B, rep, dp)-gvdwPair(r-h, beta, A, C6, B, rep, dm))/(2*h)/r;
            double rel = fabs(fd-dEdR)/max(fabs(fd), 1e-12);
            if (rel > 1e-6) {
                printf("FD mismatch rep %d r %g: analytic %.10g fd %.10g\n", rep, r, dEdR, fd);
                bad++;
            }
        }
    printf("finite differences: %s\n", bad ? "FAILED" : "ok");
    if (argc < 3)
        return bad;

    // 3. box energy with pmemd-pgm's parameters on the LJ atoms
    ifstream f(argv[1]);
    double rc = atof(argv[2]);
    int n;
    f >> n;
    double box[3][3];
    for (int a = 0; a < 3; a++)
        f >> box[a][0] >> box[a][1] >> box[a][2];
    vector<array<double, 3> > x(n);
    vector<double> R(n);
    vector<int> lj(n);
    for (int i = 0; i < n; i++)
        f >> x[i][0] >> x[i][1] >> x[i][2] >> R[i] >> lj[i];
    set<pair<int, int> > excl;
    int a, b;
    while (f >> a >> b)
        excl.insert(make_pair(min(a, b), max(a, b)));
    struct Model { const char* name; int rep; double arep, c6, b; double pmemd; };
    // pmemd-pgm VDWAALS of this box at 10 A, vdwmeth = 0 (pGM-JAX data/validation/validate_gvdw.json)
    Model models[2] = {{"slater", GVDWSlater, 87500.0, 594.825035, 4.52, 784.7282},
                       {"gauss", GVDWGaussian, 422.0, 594.825035, 0.9453, 868.3699}};
    for (const Model& m : models) {
        double A = m.arep*KCAL, C6 = m.c6*KCAL*1e-6, E = 0;
        for (int i = 0; i < n; i++) {
            if (!lj[i])
                continue;
            for (int j = i+1; j < n; j++) {
                if (!lj[j] || excl.count(make_pair(i, j)))
                    continue;
                double d[3] = {x[i][0]-x[j][0], x[i][1]-x[j][1], x[i][2]-x[j][2]};
                for (int c = 2; c >= 0; c--) {
                    double k = floor(d[c]/box[c][c]+0.5);
                    for (int q = 0; q < 3; q++)
                        d[q] -= k*box[c][q];
                }
                double r = sqrt(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]);
                if (r >= rc)
                    continue;
                double beta = 1.0/sqrt(2.0*(R[i]*R[i]+R[j]*R[j])), dEdR;
                E += gvdwPair(r, beta, A, C6, m.b, m.rep, dEdR);
            }
        }
        printf("%-6s E = %.6f kcal/mol, pmemd-pgm %.4f, diff %.2e\n", m.name, E/KCAL, m.pmemd, E/KCAL-m.pmemd);
        if (fabs(E/KCAL-m.pmemd) > 1e-3)
            bad++;
    }
    printf("%s\n", bad ? "FAILED" : "PASSED");
    return bad;
}
