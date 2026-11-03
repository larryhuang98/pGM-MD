// Recompute a GROMACS pGM dump (GMX_PGM_DUMP) with pgm_core and compare energies, forces, dipoles and
// the virial Xi = S^T/2.
//   g++ -O2 -std=c++17 -pthread -I.. ../pgm_core.cpp run_dump.cpp -o run_dump
//   ./run_dump DUMP [dipole_tol=1e-10]
#include "pgm_core.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace pgmcore;
using namespace std;

static bool nextLine(ifstream& f, string& line) {
    while (getline(f, line))
        if (!line.empty() && line[0] != '#')
            return true;
    return false;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: run_dump DUMP [dipole_tol]\n");
        return 1;
    }
    ifstream f(argv[1]);
    string line;
    nextLine(f, line);
    int n, ncov;
    Settings s;
    s.periodic = true;
    {
        istringstream is(line);
        is >> n >> ncov >> s.beta >> s.cutoff >> s.grid[0] >> s.grid[1] >> s.grid[2] >> s.order >> s.ke;
    }
    s.tol = (argc > 2 ? atof(argv[2]) : 1e-10);
    s.maxIter = 500;
    s.peek = 0;
    s.predictor = 0;
    nextLine(f, line);
    double gt[5];
    int git;
    {
        istringstream is(line);
        for (double& v : gt)
            is >> v;
        is >> git;
    }
    V3 box[3];
    double gvir[3][3];
    for (int a = 0; a < 3; a++) {
        nextLine(f, line);
        istringstream is(line);
        is >> box[a][0] >> box[a][1] >> box[a][2];
    }
    for (int a = 0; a < 3; a++) {
        nextLine(f, line);
        istringstream is(line);
        is >> gvir[a][0] >> gvir[a][1] >> gvir[a][2];
    }
    Topology top;
    vector<V3> pos(n), gF(n), gmu(n), gperm(n);
    top.q.resize(n);
    top.radius.resize(n);
    top.alpha.resize(n);
    for (int i = 0; i < n; i++) {
        nextLine(f, line);
        istringstream is(line);
        is >> pos[i][0] >> pos[i][1] >> pos[i][2] >> top.q[i] >> top.radius[i] >> top.alpha[i];
        for (int k = 0; k < 3; k++) is >> gF[i][k];
        for (int k = 0; k < 3; k++) is >> gmu[i][k];
        for (int k = 0; k < 3; k++) is >> gperm[i][k];
    }
    for (int m = 0; m < ncov; m++) {
        nextLine(f, line);
        istringstream is(line);
        int a, b;
        double c;
        is >> a >> b >> c;
        top.covAtom.push_back(a);
        top.covPartner.push_back(b);
        top.covStrength.push_back(c);
    }
    s.polarizable = false;
    for (double a : top.alpha)
        if (a > 0)
            s.polarizable = true;

    Engine eng(top, s);
    vector<V3> F(n, V3{0, 0, 0});
    double E = eng.compute(pos, box, &F);
    const EnergyTerms& T = eng.terms();
    double ct[5] = {T.direct, T.reciprocal, T.self, T.background, T.polarization};
    const char* names[5] = {"direct", "reciprocal", "self", "background", "polarization"};
    double gE = 0;
    for (int k = 0; k < 5; k++) {
        gE += gt[k];
        printf("%-13s gmx %22.12f core %22.12f diff %10.3e\n", names[k], gt[k], ct[k], gt[k]-ct[k]);
    }
    printf("%-13s gmx %22.12f core %22.12f diff %10.3e rel %10.3e\n", "total", gE, E, gE-E, (gE-E)/fabs(E));
    double dFmax = 0, F2 = 0, dmu = 0, dperm = 0;
    for (int i = 0; i < n; i++)
        for (int k = 0; k < 3; k++) {
            dFmax = max(dFmax, fabs(gF[i][k]-F[i][k]));
            F2 += F[i][k]*F[i][k];
            dmu = max(dmu, fabs(gmu[i][k]-eng.inducedDipoles()[i][k]));
            dperm = max(dperm, fabs(gperm[i][k]-eng.permanentDipoles()[i][k]));
        }
    printf("forces max|diff| %.3e (rms force %.3e);  mu max|diff| %.3e;  perm max|diff| %.3e\n", dFmax,
           sqrt(F2/(3.0*n)), dmu, dperm);
    printf("CG iterations gmx %d core %d\n", git, eng.iterations());
    const array<double, 9>& S = eng.strainDerivative();
    double dv = 0, vmax = 0;
    printf("virial Xi: gmx | core\n");
    for (int a = 0; a < 3; a++) {
        for (int b = 0; b < 3; b++) {
            double c = 0.5*S[3*b+a];
            dv = max(dv, fabs(gvir[a][b]-c));
            vmax = max(vmax, fabs(c));
        }
        printf("  %16.8f %16.8f %16.8f | %16.8f %16.8f %16.8f\n", gvir[a][0], gvir[a][1], gvir[a][2],
               0.5*S[a], 0.5*S[3+a], 0.5*S[6+a]);
    }
    printf("virial max|diff| %.3e (max|Xi| %.3e)\n", dv, vmax);
    return 0;
}
