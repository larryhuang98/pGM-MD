// pgm_cuda against pgm_core on a GROMACS pGM dump (GMX_PGM_DUMP): energies, forces, dipoles, strain with
// given pair candidates (brute force: minimum image, segments of one i atom and one shift); the engine's own
// pair list against them and after displacements that rebuild it; then the time per evaluation.
//   nvcc -O3 -std=c++17 -arch=sm_120 -I../.. ../pgm_cuda.cu ../../pgm_core.cpp test_cuda.cpp -lcufft -o test_cuda
//   ./test_cuda DUMP [tol=1e-10] [skin=0.1]          (PGM_CUDA_NO_GRAPH=1: without the CUDA graph)
#include "../pgm_cuda.h"
#include "gvdw_kernel.h"
#include "pgm_core.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

static bool nextLine(ifstream& f, string& line) {
    while (getline(f, line))
        if (!line.empty() && line[0] != '#')
            return true;
    return false;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: test_cuda DUMP [tol] [skin]\n");
        return 1;
    }
    const double tol = (argc > 2 ? atof(argv[2]) : 1e-10), skin = (argc > 3 ? atof(argv[3]) : 0.1);
    ifstream f(argv[1]);
    string line;
    nextLine(f, line);
    int n, ncov;
    pgmcore::Settings s;
    s.periodic = true;
    {
        istringstream is(line);
        is >> n >> ncov >> s.beta >> s.cutoff >> s.grid[0] >> s.grid[1] >> s.grid[2] >> s.order >> s.ke;
    }
    s.tol = tol;
    s.maxIter = 500;
    s.peek = 0;
    s.predictor = 0;
    nextLine(f, line);   // GROMACS terms (unused)
    pgmcore::V3 box[3];
    for (int a = 0; a < 3; a++) {
        nextLine(f, line);
        istringstream is(line);
        is >> box[a][0] >> box[a][1] >> box[a][2];
    }
    for (int a = 0; a < 3; a++)
        nextLine(f, line);
    pgmcore::Topology top;
    vector<pgmcore::V3> pos(n);
    top.q.resize(n);
    top.radius.resize(n);
    top.alpha.resize(n);
    for (int i = 0; i < n; i++) {
        nextLine(f, line);
        istringstream is(line);
        is >> pos[i][0] >> pos[i][1] >> pos[i][2] >> top.q[i] >> top.radius[i] >> top.alpha[i];
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

    // reference
    pgmcore::Engine ref(top, s);
    vector<pgmcore::V3> Fr(n, pgmcore::V3{0, 0, 0});
    ref.compute(pos, box, &Fr);
    const pgmcore::EnergyTerms Tr = ref.terms();   // a copy: ref is used again below

    // candidates: all pairs within cutoff + skin by the minimum image, segments of (i, shift)
    const double rl2 = (s.cutoff+skin)*(s.cutoff+skin);
    vector<int> segI, segShift, segBegin, candJ;
    for (int i = 0; i < n; i++) {
        map<array<int, 3>, vector<int>> byShift;
        for (int j = i+1; j < n; j++) {
            double d[3] = {pos[i][0]-pos[j][0], pos[i][1]-pos[j][1], pos[i][2]-pos[j][2]};
            int t[3] = {0, 0, 0};
            for (int c = 2; c >= 0; c--) {
                const double k = round(d[c]/box[c][c]);
                for (int e = 0; e < 3; e++)
                    d[e] -= k*box[c][e];
                t[c] -= (int) k;
            }
            if (d[0]*d[0]+d[1]*d[1]+d[2]*d[2] < rl2)
                byShift[{t[0], t[1], t[2]}].push_back(j);
        }
        for (auto& kv : byShift) {
            segI.push_back(i);
            segShift.insert(segShift.end(), kv.first.begin(), kv.first.end());
            segBegin.push_back(candJ.size());
            candJ.insert(candJ.end(), kv.second.begin(), kv.second.end());
        }
    }
    segBegin.push_back(candJ.size());

    pgmcuda::Settings cs;
    cs.cutoff = s.cutoff;
    cs.beta = s.beta;
    for (int d = 0; d < 3; d++)
        cs.grid[d] = s.grid[d];
    cs.order = s.order;
    cs.tol = tol;
    cs.maxIter = 500;
    cs.peek = 0;
    cs.predictor = 0;
    cs.ke = s.ke;
    pgmcuda::Engine gpu(cs);
    vector<double> pd(3*n), F(3*n, 0.0), S(9, 0.0), mu(3*n), perm(3*n);
    for (int i = 0; i < n; i++)
        for (int k = 0; k < 3; k++)
            pd[3*i+k] = pos[i][k];
    gpu.setAtoms(n, top.q.data(), top.radius.data(), top.alpha.data(), nullptr, n);
    gpu.setCovalent(ncov, top.covAtom.data(), top.covPartner.data(), top.covStrength.data());
    gpu.setCandidates(segI.size(), segI.data(), segShift.data(), segBegin.data(), candJ.data());
    double B[3][3];
    for (int a = 0; a < 3; a++)
        for (int c = 0; c < 3; c++)
            B[a][c] = box[a][c];
    pgmcuda::Terms T = gpu.compute(pd.data(), B, true, F.data(), S.data());
    gpu.inducedDipoles(mu.data());
    gpu.permanentDipoles(perm.data());

    const double ct[5] = {Tr.direct, Tr.reciprocal, Tr.self, Tr.background, Tr.polarization};
    const double gt[5] = {T.direct, T.reciprocal, T.self, T.background, T.polarization};
    const char* names[5] = {"direct", "reciprocal", "self", "background", "polarization"};
    printf("%d atoms, %zu segments, %zu candidates\n", n, segI.size(), candJ.size());
    for (int k = 0; k < 5; k++)
        printf("%-13s gpu %22.10f core %22.10f diff %10.3e\n", names[k], gt[k], ct[k], gt[k]-ct[k]);
    printf("%-13s gpu %22.10f core %22.10f rel %10.3e\n", "total", T.total(), Tr.total(),
           (T.total()-Tr.total())/fabs(Tr.total()));
    double dF = 0, F2 = 0, dmu = 0, mumax = 0, dp = 0;
    for (int i = 0; i < n; i++)
        for (int k = 0; k < 3; k++) {
            dF = max(dF, fabs(F[3*i+k]-Fr[i][k]));
            F2 += Fr[i][k]*Fr[i][k];
            dmu = max(dmu, fabs(mu[3*i+k]-ref.inducedDipoles()[i][k]));
            mumax = max(mumax, fabs(ref.inducedDipoles()[i][k]));
            dp = max(dp, fabs(perm[3*i+k]-ref.permanentDipoles()[i][k]));
        }
    printf("forces max|diff| %.3e (rms %.3e); mu max|diff| %.3e (max %.3e); perm %.3e; iterations gpu %d core %d\n",
           dF, sqrt(F2/(3.0*n)), dmu, mumax, dp, gpu.iterations(), ref.iterations());
    double dS = 0, Smax = 0;
    for (int c = 0; c < 9; c++) {
        dS = max(dS, fabs(S[c]-ref.strainDerivative()[c]));
        Smax = max(Smax, fabs(ref.strainDerivative()[c]));
    }
    printf("strain max|diff| %.3e (max %.3e)\n", dS, Smax);

    // the own pair list (cell list on the GPU) against the given candidates, then after displacements that
    // rebuild it (random moves up to 0.03 nm per component, five times)
    pgmcuda::Engine own(cs);
    own.setAtoms(n, top.q.data(), top.radius.data(), top.alpha.data(), nullptr, n);
    own.setCovalent(ncov, top.covAtom.data(), top.covPartner.data(), top.covStrength.data());
    vector<double> F2v(3*n, 0.0), S2(9, 0.0);
    pgmcuda::Terms T2 = own.compute(pd.data(), B, true, F2v.data(), S2.data());
    double dF2 = 0;
    for (int k = 0; k < 3*n; k++)
        dF2 = max(dF2, fabs(F2v[k]-F[k]));
    printf("own list: %d candidates, total %.10f (vs given candidates %.3e), forces max|diff| %.3e\n",
           own.numCandidates(), T2.total(), T2.total()-T.total(), dF2);
    bool ownOk = fabs(T2.total()-T.total()) < 1e-7*fabs(T.total());
    srand(7);
    vector<double> pm = pd;
    for (int rep = 0; rep < 5; rep++) {
        for (auto& v : pm)
            v += 0.03*(2.0*rand()/RAND_MAX-1.0);
        vector<pgmcore::V3> pv(n);
        for (int i = 0; i < n; i++)
            pv[i] = pgmcore::V3{pm[3*i], pm[3*i+1], pm[3*i+2]};
        vector<pgmcore::V3> Fc(n, pgmcore::V3{0, 0, 0});
        ref.compute(pv, box, &Fc);
        const pgmcuda::Terms Tm = own.compute(pm.data(), B, true, nullptr, nullptr);
        const double rel = (Tm.total()-ref.terms().total())/fabs(ref.terms().total());
        printf("  moved %d: list builds %ld, total rel. diff to core %.3e\n", rep+1, own.listBuilds(), rel);
        ownOk = ownOk && fabs(rel) < 1e-6;
    }
    // float positions from registered host memory (computeF): the same as from doubles
    {
        vector<float> pf(pm.begin(), pm.end());
        const pgmcuda::Terms Td = own.compute(pm.data(), B, true, nullptr, nullptr);
        const pgmcuda::Terms Tf = own.computeF(pf.data(), B, true, nullptr);
        const pgmcuda::Terms Tf2 = own.computeF(pf.data(), B, true, nullptr);
        printf("computeF: total %.10f vs %.10f from doubles (rel %.2e), again %.2e\n", Tf.total(), Td.total(),
               (Tf.total()-Td.total())/fabs(Td.total()), (Tf2.total()-Td.total())/fabs(Td.total()));
        ownOk = ownOk && fabs(Tf.total()-Td.total()) < 1e-9*fabs(Td.total());
    }

    // the same atoms in another order (as after a GROMACS pair search): the list is renamed, not rebuilt
    {
        vector<int> perm(n), gidx(n);   // new local k holds old atom perm[k]
        for (int k = 0; k < n; k++)
            perm[k] = (k*7919+13)%n;
        vector<int> used(n, 0);
        bool bij = true;
        for (int k = 0; k < n; k++)
            bij = bij && !used[perm[k]]++;
        if (!bij)
            for (int k = 0; k < n; k++)
                perm[k] = n-1-k;
        vector<int> newOf(n);
        vector<double> q2(n), r2v(n), a2(n), p2(3*n);
        for (int k = 0; k < n; k++) {
            newOf[perm[k]] = k;
            q2[k] = top.q[perm[k]];
            r2v[k] = top.radius[perm[k]];
            a2[k] = top.alpha[perm[k]];
            gidx[k] = perm[k];
            for (int c = 0; c < 3; c++)
                p2[3*k+c] = pm[3*perm[k]+c];
        }
        vector<int> ca(ncov), cp(ncov);
        for (int m = 0; m < ncov; m++) {
            ca[m] = newOf[top.covAtom[m]];
            cp[m] = newOf[top.covPartner[m]];
        }
        const pgmcuda::Terms Ta = own.compute(pm.data(), B, true, F2v.data(), nullptr);
        const long builds0 = own.listBuilds();
        own.setAtoms(n, q2.data(), r2v.data(), a2.data(), gidx.data(), n);
        own.setCovalent(ncov, ca.data(), cp.data(), top.covStrength.data());
        vector<double> Fp(3*n, 0.0);
        const pgmcuda::Terms Tb = own.compute(p2.data(), B, true, Fp.data(), nullptr);
        printf("reordered: total %.10f vs %.10f (rel %.2e), list builds %ld -> %ld\n", Tb.total(), Ta.total(),
               (Tb.total()-Ta.total())/fabs(Ta.total()), builds0, own.listBuilds());
        ownOk = ownOk && fabs(Tb.total()-Ta.total()) < 1e-7*fabs(Ta.total()) && own.listBuilds() == builds0;
    }
    own.setAtoms(n, top.q.data(), top.radius.data(), top.alpha.data(), nullptr, n);
    own.setCovalent(ncov, top.covAtom.data(), top.covPartner.data(), top.covStrength.data());

    // GVDW (Slater), water-like parameters (negative charges: O, else H), the covalent pairs skipped:
    // the engine with GVDW minus without against a double brute-force sum
    {
        vector<double> ga(n), gc(n), gb(n);
        for (int i = 0; i < n; i++) {
            const bool O = top.q[i] < 0;
            ga[i] = O ? 605.06 : 20.0;
            gc[i] = O ? 0.0499 : 0.005;
            gb[i] = O ? 4.52 : 4.0;
        }
        set<pair<int, int>> skip;
        for (int m = 0; m < ncov; m++)
            skip.insert({min(top.covAtom[m], top.covPartner[m]), max(top.covAtom[m], top.covPartner[m])});
        vector<int> si, sj;
        for (auto& pr : skip) {
            si.push_back(pr.first);
            sj.push_back(pr.second);
        }
        const double rv = s.cutoff;
        double Eref = 0, Sref[9] = {0};
        vector<double> Fref(3*n, 0.0);
        for (int i = 0; i < n; i++)
            for (int j = i+1; j < n; j++) {
                if (skip.count({i, j}))
                    continue;
                double d[3] = {pos[i][0]-pos[j][0], pos[i][1]-pos[j][1], pos[i][2]-pos[j][2]};
                for (int c = 2; c >= 0; c--) {
                    const double k = round(d[c]/box[c][c]);
                    for (int e = 0; e < 3; e++)
                        d[e] -= k*box[c][e];
                }
                const double r = sqrt(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]);
                if (r >= rv)
                    continue;
                const double bij = 1.0/sqrt(2.0*(top.radius[i]*top.radius[i]+top.radius[j]*top.radius[j]));
                double dEdR;
                Eref += pgmcore::gvdwPair(r, bij, ga[i]*ga[j], gc[i]*gc[j], 0.5*(gb[i]+gb[j]), pgmcore::GVDWSlater, dEdR);
                for (int a = 0; a < 3; a++) {
                    Fref[3*i+a] += dEdR*d[a];
                    Fref[3*j+a] -= dEdR*d[a];
                    for (int b = 0; b < 3; b++)
                        Sref[3*a+b] -= dEdR*d[a]*d[b];
                }
            }
        pgmcuda::Engine gv(cs);
        gv.setAtoms(n, top.q.data(), top.radius.data(), top.alpha.data(), nullptr, n);
        gv.setCovalent(ncov, top.covAtom.data(), top.covPartner.data(), top.covStrength.data());
        vector<double> F0(3*n, 0.0), S0(9, 0.0), F1(3*n, 0.0), S1(9, 0.0);
        const pgmcuda::Terms T0 = gv.compute(pd.data(), B, true, F0.data(), S0.data());
        gv.setAtoms(n, top.q.data(), top.radius.data(), top.alpha.data(), nullptr, n);
        gv.setCovalent(ncov, top.covAtom.data(), top.covPartner.data(), top.covStrength.data());
        gv.setGvdw(1, rv, ga.data(), gc.data(), gb.data(), (int) si.size(), si.data(), sj.data());
        const pgmcuda::Terms T1 = gv.compute(pd.data(), B, true, F1.data(), S1.data());
        double dFg = 0, Fgmax = 0, dSg = 0, Sgmax = 0;
        for (int k = 0; k < 3*n; k++) {
            dFg = max(dFg, fabs(F1[k]-F0[k]-Fref[k]));
            Fgmax = max(Fgmax, fabs(Fref[k]));
        }
        for (int c = 0; c < 9; c++) {
            dSg = max(dSg, fabs(S1[c]-S0[c]-Sref[c]));
            Sgmax = max(Sgmax, fabs(Sref[c]));
        }
        const double relE = (T1.gvdw-Eref)/fabs(Eref);
        printf("GVDW: gpu %.8f ref %.8f rel %.3e (electrostatics change %.3e); forces max|diff| %.3e (max %.3e); "
               "strain max|diff| %.3e (max %.3e)\n", T1.gvdw, Eref, relE, (T1.total()-T1.gvdw)-T0.total(), dFg,
               Fgmax, dSg, Sgmax);
        ownOk = ownOk && fabs(relE) < 1e-5 && dFg < 1e-4*Fgmax && dSg < 1e-4*Sgmax;
    }

    // with the mu4 predictor over a few moved steps (the dual first field pass; PGM_CUDA_NO_DUAL=1 for the
    // separate passes: the same numbers to rounding) against pgm_core at each step
    {
        pgmcuda::Settings ps = cs;
        ps.predictor = 2;
        pgmcuda::Engine pe(ps);
        pe.setAtoms(n, top.q.data(), top.radius.data(), top.alpha.data(), nullptr, n);
        pe.setCovalent(ncov, top.covAtom.data(), top.covPartner.data(), top.covStrength.data());
        srand(11);
        vector<double> pm2 = pd;
        double worst = 0;
        for (int step = 0; step < 6; step++) {
            for (auto& v : pm2)
                v += 0.002*(2.0*rand()/RAND_MAX-1.0);
            vector<pgmcore::V3> pv(n);
            for (int i = 0; i < n; i++)
                pv[i] = pgmcore::V3{pm2[3*i], pm2[3*i+1], pm2[3*i+2]};
            vector<pgmcore::V3> Fc(n, pgmcore::V3{0, 0, 0});
            ref.compute(pv, box, &Fc);
            const pgmcuda::Terms Tp = pe.compute(pm2.data(), B, true, nullptr, nullptr);
            const double rel = (Tp.total()-ref.terms().total())/fabs(ref.terms().total());
            worst = max(worst, fabs(rel));
            printf("  predictor step %d: total %.10f, %d iterations, rel. diff to core %.3e\n", step, Tp.total(),
                   pe.iterations(), rel);
        }
        ownOk = ownOk && worst < 1e-6;
    }

    // timing: repeated evaluations at tolerance 1e-5 with the predictor (same positions: iterations ~0)
    const int reps = 50;
    auto t0 = chrono::steady_clock::now();
    for (int r = 0; r < reps; r++)
        own.compute(pd.data(), B, true, F.data(), nullptr);
    const double ms = 1e3*chrono::duration<double>(chrono::steady_clock::now()-t0).count()/reps;
    printf("timing (own list): %.3f ms per evaluation (%d iterations, tol %.1e)\n", ms, own.iterations(), tol);
    const bool ok = fabs(T.total()-Tr.total()) < 1e-6*fabs(Tr.total()) && dF < 1e-3*sqrt(F2/(3.0*n))*100 && ownOk;
    printf("%s\n", ok ? "PASSED" : "CHECK");
    return ok ? 0 : 1;
}
