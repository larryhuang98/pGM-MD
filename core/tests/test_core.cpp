// Finite-difference checks of pgm_core: forces and the strain derivative (virial) for random water-like
// molecules in a triclinic box (PME) and without a box, all levels on (charges, covalent dipoles,
// induced dipoles solved tightly).
//   g++ -O2 -std=c++17 -I.. ../pgm_core.cpp test_core.cpp -o test_core && ./test_core
#include "pgm_core.h"
#include <cmath>
#include <cstdio>
#include <random>

using namespace pgmcore;

static void build(int nmol, unsigned seed, const V3 box[3], std::vector<V3>& pos, Topology& top) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    for (int m = 0; m < nmol; m++) {
        V3 o;
        double f[3] = {U(rng), U(rng), U(rng)};
        for (int k = 0; k < 3; k++)
            o[k] = f[0]*box[0][k]+f[1]*box[1][k]+f[2]*box[2][k];
        int base = pos.size();
        pos.push_back(o);
        for (int h = 0; h < 2; h++) {
            double th = 2*M_PI*U(rng), ph = acos(2*U(rng)-1);
            pos.push_back(V3{o[0]+0.096*sin(ph)*cos(th), o[1]+0.096*sin(ph)*sin(th), o[2]+0.096*cos(ph)});
        }
        top.q.insert(top.q.end(), {-0.8, 0.4, 0.4});
        top.radius.insert(top.radius.end(), {0.0605, 0.0536, 0.0536});
        top.alpha.insert(top.alpha.end(), {1.1e-3, 0.33e-3, 0.33e-3});
        for (int h = 1; h <= 2; h++) {
            top.covAtom.push_back(base);
            top.covPartner.push_back(base+h);
            top.covStrength.push_back(-0.019);
            top.covAtom.push_back(base+h);
            top.covPartner.push_back(base);
            top.covStrength.push_back(0.0086);
        }
    }
}

int main() {
    int fails = 0;
    for (int periodic = 1; periodic >= 0; periodic--) {
        V3 box[3] = {{2.0, 0, 0}, {-0.5, 1.9, 0}, {-0.4, -0.7, 1.8}};
        std::vector<V3> pos;
        Topology top;
        build(periodic ? 24 : 10, 7, box, pos, top);
        Settings s;
        s.periodic = periodic;
        s.cutoff = 0.85;
        s.beta = 5.0;              // erfc(beta rc) ~ 1e-9: no cutoff discontinuity in the differences
        s.grid[0] = s.grid[1] = s.grid[2] = 48;
        s.order = 6;
        s.tol = 1e-13;
        s.maxIter = 2000;
        s.peek = 0;
        s.predictor = 0;
        Engine eng(top, s);
        std::vector<V3> F(pos.size(), V3{0, 0, 0});
        double e0 = eng.compute(pos, box, &F);
        std::array<double, 9> S = eng.strainDerivative();
        printf("%s: %zu atoms, E = %.8f kJ/mol, %d CG iterations\n", periodic ? "PME" : "no cutoff", pos.size(), e0, eng.iterations());
        // forces
        double h = 1e-5, maxErr = 0, maxF = 0;
        for (int i : {0, 1, 2, 7, 20}) {
            for (int k = 0; k < 3; k++) {
                std::vector<V3> p = pos;
                p[i][k] += h;
                Engine e1(top, s);
                double ep = e1.compute(p, box, nullptr);
                p[i][k] -= 2*h;
                Engine e2(top, s);
                double em = e2.compute(p, box, nullptr);
                double fd = -(ep-em)/(2*h);
                maxErr = std::max(maxErr, fabs(fd-F[i][k]));
                maxF = std::max(maxF, fabs(F[i][k]));
            }
        }
        printf("  forces: max |F_fd - F| = %.3e (max |F| %.1f)\n", maxErr, maxF);
        if (maxErr > 1e-5*maxF)
            fails++;
        if (!periodic)
            continue;
        // strain derivative, upper-triangular components (the box stays in reduced form)
        double hs = 1e-6, maxS = 0, errS = 0;
        for (int a = 0; a < 3; a++)
            for (int b = a; b < 3; b++) {
                double ef[2];
                for (int sgn = 0; sgn < 2; sgn++) {
                    double eps = (sgn == 0 ? hs : -hs);
                    std::vector<V3> p = pos;
                    for (auto& r : p)
                        r[a] += eps*r[b];
                    V3 bx[3] = {box[0], box[1], box[2]};
                    for (auto& v : bx)
                        v[a] += eps*v[b];
                    Engine e1(top, s);
                    ef[sgn] = e1.compute(p, bx, nullptr);
                }
                double fd = (ef[0]-ef[1])/(2*hs);
                printf("  S[%d%d] analytic %.6f  finite difference %.6f\n", a, b, S[3*a+b], fd);
                errS = std::max(errS, fabs(fd-S[3*a+b]));
                maxS = std::max(maxS, fabs(S[3*a+b]));
            }
        printf("  symmetry: S01-S10 %.2e S02-S20 %.2e S12-S21 %.2e\n", S[1]-S[3], S[2]-S[6], S[5]-S[7]);
        if (errS > 1e-5*maxS)
            fails++;
    }
    printf(fails ? "FAILED\n" : "PASSED\n");
    return fails;
}
