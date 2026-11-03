#include "pgm_core.h"
#include "pgm_constants.h"
#include "pocketfft_hdronly.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>

using namespace std;

namespace pgmcore {

static const double SQRT_PI = 1.7724538509055160273;

static inline double dot(const V3& a, const V3& b) {
    return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
}

void erfKernels(double a, double r, int nmax, double* B) {
    double x = a*r, s = x*x;
    double c0 = 2.0*a/SQRT_PI, a2 = 2.0*a*a;
    if (s < 1.0) {
        // B_n = c0 (2a^2)^n sum_m (-s)^m / (m! (2m+2n+1)), m = 0..14
        double t[15];
        t[0] = 1.0;
        for (int m = 1; m < 15; m++)
            t[m] = t[m-1]*(-s)/m;
        double pw = 1.0;
        for (int k = 0; k < nmax; k++) {
            double sum = 0.0;
            for (int m = 14; m >= 0; m--)
                sum += t[m]/(2*m+2*k+1);
            B[k] = c0*pw*sum;
            pw *= a2;
        }
    }
    else {
        double c = c0*exp(-s), ir2 = 1.0/(r*r);
        B[0] = erf(x)/r;
        double pw = 1.0;
        for (int k = 1; k < nmax; k++) {
            B[k] = ((2*k-1)*B[k-1]-pw*c)*ir2;
            pw *= a2;
        }
    }
}

static void applyPolarizabilityThreshold(Topology& top) {
    for (double& a : top.alpha)
        if (a <= MIN_POLARIZABILITY)
            a = 0.0;   // not polarizable (pmemd-pgm's threshold)
}

Engine::Engine(const Topology& topology, const Settings& settings) : top(topology), s(settings) {
    applyPolarizabilityThreshold(top);
    n = top.q.size();
    if ((int) top.radius.size() != n || (int) top.alpha.size() != n)
        throw invalid_argument("pgmcore: per-atom arrays of different lengths");
    perm.assign(n, V3{0, 0, 0});
    mu.assign(n, V3{0, 0, 0});
    if (s.periodic) {
        for (int a = 0; a < 3; a++)
            if (s.grid[a] <= 0)
                throw invalid_argument("pgmcore: PME grid not set");
        if (s.order < 3)
            throw invalid_argument("pgmcore: PME order must be >= 3");
        if (s.beta <= 0)
            throw invalid_argument("pgmcore: Ewald coefficient not set");
    }
}

void Engine::setTopology(const Topology& t) {
    if ((int) t.q.size() != n)
        throw invalid_argument("pgmcore: setTopology cannot change the number of atoms");
    top = t;
    applyPolarizabilityThreshold(top);
    havePos = false;
}

// ------------------------------------------------------------------------- geometry

static void minImage(V3& x, const V3 box[3]) {
    for (int c = 2; c >= 0; c--) {
        double k = round(x[c]/box[c][c]);
        for (int j = 0; j < 3; j++)
            x[j] -= k*box[c][j];
    }
}

void Engine::perm_dipoles() {
    perm.assign(n, V3{0, 0, 0});
    for (size_t m = 0; m < top.covAtom.size(); m++) {
        int i = top.covAtom[m], j = top.covPartner[m];
        V3 v = {pos[j][0]-pos[i][0], pos[j][1]-pos[i][1], pos[j][2]-pos[i][2]};
        if (s.periodic)
            minImage(v, box);
        double L = sqrt(dot(v, v));
        for (int k = 0; k < 3; k++)
            perm[i][k] += top.covStrength[m]*v[k]/L;
    }
}

void Engine::build_pairs() {
    pairs.clear();
    double rc2 = s.cutoff*s.cutoff;
    double Bb[4];
    for (int i = 0; i < n; i++)
        for (int j = i+1; j < n; j++) {
            Pair p;
            p.i = i;
            p.j = j;
            for (int k = 0; k < 3; k++)
                p.x[k] = pos[i][k]-pos[j][k];
            if (s.periodic) {
                minImage(p.x, box);
                if (dot(p.x, p.x) >= rc2)
                    continue;
            }
            double r = sqrt(dot(p.x, p.x));
            double a = 1.0/sqrt(2.0*(top.radius[i]*top.radius[i]+top.radius[j]*top.radius[j]));
            erfKernels(a, r, 4, p.G);
            if (s.periodic) {
                erfKernels(s.beta, r, 4, Bb);
                for (int k = 0; k < 4; k++)
                    p.G[k] -= Bb[k];
            }
            pairs.push_back(p);
        }
}

void Engine::setGeometry(const vector<V3>& positions, const V3 boxIn[3]) {
    pos = positions;
    if (s.periodic) {
        for (int i = 0; i < 3; i++)
            box[i] = boxIn[i];
        // H has the box vectors as rows; recip = H^-1
        const V3* H = box;
        double det = H[0][0]*(H[1][1]*H[2][2]-H[1][2]*H[2][1])-H[0][1]*(H[1][0]*H[2][2]-H[1][2]*H[2][0])
                    +H[0][2]*(H[1][0]*H[2][1]-H[1][1]*H[2][0]);
        volume = fabs(det);
        double inv[3][3];
        inv[0][0] = (H[1][1]*H[2][2]-H[1][2]*H[2][1])/det;
        inv[0][1] = (H[0][2]*H[2][1]-H[0][1]*H[2][2])/det;
        inv[0][2] = (H[0][1]*H[1][2]-H[0][2]*H[1][1])/det;
        inv[1][0] = (H[1][2]*H[2][0]-H[1][0]*H[2][2])/det;
        inv[1][1] = (H[0][0]*H[2][2]-H[0][2]*H[2][0])/det;
        inv[1][2] = (H[0][2]*H[1][0]-H[0][0]*H[1][2])/det;
        inv[2][0] = (H[1][0]*H[2][1]-H[1][1]*H[2][0])/det;
        inv[2][1] = (H[0][1]*H[2][0]-H[0][0]*H[2][1])/det;
        inv[2][2] = (H[0][0]*H[1][1]-H[0][1]*H[1][0])/det;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                recip[i][j] = inv[i][j];
    }
    perm_dipoles();
    build_pairs();
    if (s.periodic) {
        setup_pme();
        compute_influence();
    }
}

// ------------------------------------------------------------------------- PME

void Engine::setup_pme() {
    int p = s.order;
    for (int a = 0; a < 3; a++) {
        gridIndex[a].assign(n, 0);
        theta[a].assign(n*p, 0.0);
        dtheta[a].assign(n*p, 0.0);
        d2theta[a].assign(n*p, 0.0);
    }
    // M[k][j] = M_k(fr + j) for j = 0..k-1 (cardinal B-spline of order k), zero outside
    vector<vector<double> > M(p+1, vector<double>(p+2, 0.0));
    auto get = [&](int k, int j) { return (j >= 0 && j < k) ? M[k][j] : 0.0; };
    for (int i = 0; i < n; i++)
        for (int a = 0; a < 3; a++) {
            double t = pos[i][0]*recip[0][a]+pos[i][1]*recip[1][a]+pos[i][2]*recip[2][a];
            t -= floor(t);
            double w = s.grid[a]*t;
            int base = (int) floor(w);
            double fr = w-base;
            M[1][0] = 1.0;
            for (int k = 2; k <= p; k++)
                for (int j = 0; j < k; j++)
                    M[k][j] = ((fr+j)*get(k-1, j)+(k-fr-j)*get(k-1, j-1))/(k-1);
            // M_p'(u) = M_{p-1}(u) - M_{p-1}(u-1);  M_p''(u) = M_{p-2}(u) - 2 M_{p-2}(u-1) + M_{p-2}(u-2)
            for (int j = 0; j < p; j++) {
                theta[a][i*p+j] = M[p][j];
                dtheta[a][i*p+j] = get(p-1, j)-get(p-1, j-1);
                d2theta[a][i*p+j] = get(p-2, j)-2*get(p-2, j-1)+get(p-2, j-2);
            }
            gridIndex[a][i] = base;
        }
}

static vector<double> bsplineModuli(int K, int p) {
    // |sum_k M_p(k+1) exp(2 pi i m k / K)|^2, zeros replaced by the mean of the neighbours (OpenMM)
    vector<double> M(p+1, 0.0), tmp(p+1, 0.0);
    M[0] = 1.0;   // M_1 at 0 (fr = 0): M_k(j) for j = 0..k-1
    for (int k = 2; k <= p; k++) {
        for (int j = 0; j < k; j++) {
            double cur = (j < k-1 ? M[j] : 0.0), prev = (j > 0 ? M[j-1] : 0.0);
            tmp[j] = (j*cur+(k-j)*prev)/(k-1);
        }
        for (int j = 0; j < k; j++)
            M[j] = tmp[j];
    }
    vector<double> data(K, 0.0);
    for (int k = 0; k < p-1 && k < K; k++)
        data[k] = M[k+1];
    vector<double> mod(K);
    for (int m = 0; m < K; m++) {
        double sc = 0, ss = 0;
        for (int k = 0; k < K; k++) {
            double arg = 2*M_PI*m*k/K;
            sc += data[k]*cos(arg);
            ss += data[k]*sin(arg);
        }
        mod[m] = sc*sc+ss*ss;
    }
    for (int m = 0; m < K; m++)
        if (mod[m] < 1e-7)
            mod[m] = 0.5*(mod[(m-1+K)%K]+mod[(m+1)%K]);
    return mod;
}

void Engine::compute_influence() {
    int K1 = s.grid[0], K2 = s.grid[1], K3 = s.grid[2], K3h = K3/2+1;
    vector<double> b1 = bsplineModuli(K1, s.order), b2 = bsplineModuli(K2, s.order), b3 = bsplineModuli(K3, s.order);
    influence.assign((size_t) K1*K2*K3h, 0.0);
    double fac = M_PI*M_PI/(s.beta*s.beta);
    for (int m1 = 0; m1 < K1; m1++) {
        int mm1 = (m1 <= K1/2 ? m1 : m1-K1);
        for (int m2 = 0; m2 < K2; m2++) {
            int mm2 = (m2 <= K2/2 ? m2 : m2-K2);
            for (int m3 = 0; m3 < K3h; m3++) {
                int mm3 = m3;
                if (mm1 == 0 && mm2 == 0 && mm3 == 0)
                    continue;
                V3 ms;
                for (int k = 0; k < 3; k++)
                    ms[k] = mm1*recip[k][0]+mm2*recip[k][1]+mm3*recip[k][2];
                double m2s = dot(ms, ms);
                double g = exp(-fac*m2s)/(M_PI*volume*m2s)/(b1[m1]*b2[m2]*b3[m3]);
                influence[((size_t) m1*K2+m2)*K3h+m3] = g;
            }
        }
    }
}

vector<V3> Engine::frac_dipoles(const vector<V3>& d) const {
    vector<V3> e(n);
    for (int i = 0; i < n; i++)
        for (int a = 0; a < 3; a++)
            e[i][a] = s.grid[a]*(d[i][0]*recip[0][a]+d[i][1]*recip[1][a]+d[i][2]*recip[2][a]);
    return e;
}

double Engine::pme_potential(const vector<double>* q, const vector<V3>& d, vector<double>& phi, double* strain) {
    int K1 = s.grid[0], K2 = s.grid[1], K3 = s.grid[2], K3h = K3/2+1, p = s.order;
    vector<double> Q((size_t) K1*K2*K3, 0.0);
    vector<V3> e = frac_dipoles(d);
    for (int i = 0; i < n; i++) {
        double qi = (q ? (*q)[i] : 0.0);
        for (int j1 = 0; j1 < p; j1++) {
            int g1 = ((gridIndex[0][i]-j1)%K1+K1)%K1;
            double t1 = theta[0][i*p+j1], u1 = dtheta[0][i*p+j1];
            for (int j2 = 0; j2 < p; j2++) {
                int g2 = ((gridIndex[1][i]-j2)%K2+K2)%K2;
                double t2 = theta[1][i*p+j2], u2 = dtheta[1][i*p+j2];
                for (int j3 = 0; j3 < p; j3++) {
                    int g3 = ((gridIndex[2][i]-j3)%K3+K3)%K3;
                    double t3 = theta[2][i*p+j3], u3 = dtheta[2][i*p+j3];
                    double w = qi*t1*t2*t3+e[i][0]*u1*t2*t3+e[i][1]*t1*u2*t3+e[i][2]*t1*t2*u3;
                    Q[((size_t) g1*K2+g2)*K3+g3] += w;
                }
            }
        }
    }
    vector<complex<double> > F((size_t) K1*K2*K3h);
    pocketfft::shape_t shape = {(size_t) K1, (size_t) K2, (size_t) K3};
    pocketfft::stride_t sr = {(ptrdiff_t) (sizeof(double)*K2*K3), (ptrdiff_t) (sizeof(double)*K3), (ptrdiff_t) sizeof(double)};
    pocketfft::stride_t sc = {(ptrdiff_t) (sizeof(complex<double>)*K2*K3h), (ptrdiff_t) (sizeof(complex<double>)*K3h), (ptrdiff_t) sizeof(complex<double>)};
    pocketfft::shape_t axes = {0, 1, 2};
    pocketfft::r2c(shape, sr, sc, axes, pocketfft::FORWARD, Q.data(), F.data(), 1.0);
    double energy = 0.0;
    for (int m1 = 0; m1 < K1; m1++)
        for (int m2 = 0; m2 < K2; m2++)
            for (int m3 = 0; m3 < K3h; m3++) {
                size_t idx = ((size_t) m1*K2+m2)*K3h+m3;
                double g = influence[idx];
                double w = (m3 == 0 || (K3%2 == 0 && m3 == K3/2) ? 1.0 : 2.0);
                energy += 0.5*w*g*norm(F[idx]);
                if (strain && g != 0) {
                    int mm1 = (m1 <= K1/2 ? m1 : m1-K1), mm2 = (m2 <= K2/2 ? m2 : m2-K2), mm3 = m3;
                    V3 ms;
                    for (int k = 0; k < 3; k++)
                        ms[k] = mm1*recip[k][0]+mm2*recip[k][1]+mm3*recip[k][2];
                    double m2s = dot(ms, ms);
                    double e = 0.5*w*g*norm(F[idx]);
                    double c2 = 2.0*(M_PI*M_PI/(s.beta*s.beta)+1.0/m2s);
                    for (int a = 0; a < 3; a++)
                        for (int b = 0; b < 3; b++)
                            strain[3*a+b] += e*((a == b ? -1.0 : 0.0)+c2*ms[a]*ms[b]);
                }
                F[idx] *= g;
            }
    phi.assign((size_t) K1*K2*K3, 0.0);
    pocketfft::c2r(shape, sc, sr, axes, pocketfft::BACKWARD, F.data(), phi.data(), 1.0);
    return energy;
}

void Engine::pme_field(const vector<double>& phi, vector<V3>& E) const {
    int K1 = s.grid[0], K2 = s.grid[1], K3 = s.grid[2], p = s.order;
    for (int i = 0; i < n; i++) {
        double f[3] = {0, 0, 0};
        for (int j1 = 0; j1 < p; j1++) {
            int g1 = ((gridIndex[0][i]-j1)%K1+K1)%K1;
            double t1 = theta[0][i*p+j1], u1 = dtheta[0][i*p+j1];
            for (int j2 = 0; j2 < p; j2++) {
                int g2 = ((gridIndex[1][i]-j2)%K2+K2)%K2;
                double t2 = theta[1][i*p+j2], u2 = dtheta[1][i*p+j2];
                for (int j3 = 0; j3 < p; j3++) {
                    int g3 = ((gridIndex[2][i]-j3)%K3+K3)%K3;
                    double t3 = theta[2][i*p+j3], u3 = dtheta[2][i*p+j3];
                    double ph = phi[((size_t) g1*K2+g2)*K3+g3];
                    f[0] += u1*t2*t3*ph;
                    f[1] += t1*u2*t3*ph;
                    f[2] += t1*t2*u3*ph;
                }
            }
        }
        for (int k = 0; k < 3; k++)
            E[i][k] -= s.grid[0]*recip[k][0]*f[0]+s.grid[1]*recip[k][1]*f[1]+s.grid[2]*recip[k][2]*f[2];
    }
}

// ------------------------------------------------------------------------- fields and solver

vector<V3> Engine::field(const vector<double>& q, const vector<V3>& d) {
    vector<V3> E(n, V3{0, 0, 0});
    bool useq = false;
    for (double v : q)
        if (v != 0)
            useq = true;
    for (const Pair& p : pairs) {
        const V3& x = p.x;
        double qi = q[p.i], qj = q[p.j];
        const V3& di = d[p.i];
        const V3& dj = d[p.j];
        double djx = dot(dj, x), dix = dot(di, x);
        for (int k = 0; k < 3; k++) {
            E[p.i][k] += qj*x[k]*p.G[1]-dj[k]*p.G[1]+djx*x[k]*p.G[2];
            E[p.j][k] += -qi*x[k]*p.G[1]-di[k]*p.G[1]+dix*x[k]*p.G[2];
        }
    }
    if (s.periodic) {
        vector<double> phi;
        pme_potential(useq ? &q : nullptr, d, phi);
        pme_field(phi, E);
        double cs = 4.0*s.beta*s.beta*s.beta/(3.0*SQRT_PI);
        for (int i = 0; i < n; i++)
            for (int k = 0; k < 3; k++)
                E[i][k] += cs*d[i][k];
    }
    return E;
}

void Engine::solve() {
    vector<double> zero(n, 0.0);
    vector<V3> b = field(top.q, perm);
    vector<double> al = top.alpha;
    double norm = 0;
    for (int i = 0; i < n; i++)
        for (int k = 0; k < 3; k++)
            norm += fabs(al[i]*b[i][k]);
    norm = norm/(3.0*n)+1e-300;

    // initial guess
    vector<V3> x(n, V3{0, 0, 0});
    bool samePos = havePos && (historyKey >= 0 ? historyKey == lastKey : lastPos == pos);
    int h = history.size();
    if (samePos && h > 0)
        x = history[0];
    else if (s.predictor == 2 && h >= 4) {
        for (int i = 0; i < n; i++)
            for (int k = 0; k < 3; k++)
                x[i][k] = 4*history[0][i][k]-6*history[1][i][k]+4*history[2][i][k]-history[3][i][k];
    }
    else if (s.predictor >= 1 && h >= 3) {
        for (int i = 0; i < n; i++)
            for (int k = 0; k < 3; k++)
                x[i][k] = 3*history[0][i][k]-3*history[1][i][k]+history[2][i][k];
    }
    else if (s.predictor >= 1 && h >= 1)
        x = history[0];
    else
        for (int i = 0; i < n; i++)
            for (int k = 0; k < 3; k++)
                x[i][k] = al[i]*b[i][k];

    auto applyA = [&](const vector<V3>& v) {   // alpha^-1 v - T v (rows of polarizable atoms)
        vector<V3> Tv = field(zero, v);
        vector<V3> out(n);
        for (int i = 0; i < n; i++)
            for (int k = 0; k < 3; k++)
                out[i][k] = (al[i] > 0 ? v[i][k]/al[i]-Tv[i][k] : 0.0);
        return out;
    };
    auto vdot = [&](const vector<V3>& a, const vector<V3>& c) {
        double sum = 0;
        for (int i = 0; i < n; i++)
            sum += dot(a[i], c[i]);
        return sum;
    };
    auto errOf = [&](const vector<V3>& r) {
        double m = 0;
        for (int i = 0; i < n; i++)
            for (int k = 0; k < 3; k++)
                m = max(m, fabs(al[i]*r[i][k]));
        return m/norm;
    };
    // r = b - A x
    vector<V3> Ax = applyA(x), r(n), z(n), pdir(n);
    for (int i = 0; i < n; i++)
        for (int k = 0; k < 3; k++) {
            r[i][k] = (al[i] > 0 ? b[i][k]-Ax[i][k] : 0.0);
            z[i][k] = al[i]*r[i][k];
        }
    pdir = z;
    double rz = vdot(r, z);
    double err = errOf(r);
    int it = 0;
    while (err > s.tol && it < s.maxIter) {
        vector<V3> Ap = applyA(pdir);
        double alph = rz/max(vdot(pdir, Ap), 1e-300);
        vector<V3> rnew(n), znew(n);
        for (int i = 0; i < n; i++)
            for (int k = 0; k < 3; k++) {
                x[i][k] += alph*pdir[i][k];
                rnew[i][k] = r[i][k]-alph*Ap[i][k];
                znew[i][k] = al[i]*rnew[i][k];
            }
        double beta = 0;
        for (int i = 0; i < n; i++)
            for (int k = 0; k < 3; k++)
                beta += znew[i][k]*(rnew[i][k]-r[i][k]);
        beta /= max(rz, 1e-300);
        for (int i = 0; i < n; i++)
            for (int k = 0; k < 3; k++)
                pdir[i][k] = znew[i][k]+beta*pdir[i][k];
        r = rnew;
        z = znew;
        rz = vdot(r, z);
        err = errOf(r);
        it++;
    }
    if (s.peek != 0)
        for (int i = 0; i < n; i++)
            for (int k = 0; k < 3; k++)
                x[i][k] += s.peek*al[i]*r[i][k];
    mu = x;
    lastIterations = it;
    lastResidual = err;
    if (!samePos) {
        history.push_front(mu);
        while (history.size() > 4)
            history.pop_back();
    }
    lastPos = pos;
    lastKey = historyKey;
    havePos = true;
}

// ------------------------------------------------------------------------- energy and forces

double Engine::compute(const vector<V3>& positions, const V3 boxIn[3], vector<V3>* forces) {
    setGeometry(positions, boxIn);
    if (s.polarizable)
        solve();
    else {
        mu.assign(n, V3{0, 0, 0});
        lastIterations = 0;
    }
    vector<V3> d(n);
    for (int i = 0; i < n; i++)
        for (int k = 0; k < 3; k++)
            d[i][k] = perm[i][k]+mu[i][k];
    const vector<double>& q = top.q;
    EnergyTerms T;
    vector<V3> E(n, V3{0, 0, 0});        // total field at fixed d [e/nm^2]
    vector<V3> F(n, V3{0, 0, 0});        // forces at fixed d [e^2/nm^2]
    double S[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};   // strain derivative [e^2/nm]
    for (const Pair& p : pairs) {
        const V3& x = p.x;
        double qi = q[p.i], qj = q[p.j];
        const V3& di = d[p.i];
        const V3& dj = d[p.j];
        double dix = dot(di, x), djx = dot(dj, x), didj = dot(di, dj);
        const double* G = p.G;
        T.direct += qi*qj*G[0]+(qi*djx-qj*dix)*G[1]+didj*G[1]-dix*djx*G[2];
        double sc = -qi*qj*G[1]-(qi*djx-qj*dix)*G[2]-didj*G[2]+dix*djx*G[3];
        for (int k = 0; k < 3; k++) {
            double dedx = sc*x[k]+(qi*dj[k]-qj*di[k])*G[1]-(djx*di[k]+dix*dj[k])*G[2];
            for (int b = 0; b < 3; b++)
                S[3*k+b] += dedx*x[b];
            F[p.i][k] -= dedx;
            F[p.j][k] += dedx;
            E[p.i][k] += qj*x[k]*G[1]-dj[k]*G[1]+djx*x[k]*G[2];
            E[p.j][k] += -qi*x[k]*G[1]-di[k]*G[1]+dix*x[k]*G[2];
        }
    }
    if (s.periodic) {
        vector<double> phi;
        T.reciprocal = pme_potential(&q, d, phi, S);
        vector<V3> Erec(n, V3{0, 0, 0});
        pme_field(phi, Erec);
        for (int i = 0; i < n; i++)
            for (int a = 0; a < 3; a++) {
                E[i][a] += Erec[i][a];
                for (int b = 0; b < 3; b++)
                    S[3*a+b] += Erec[i][a]*d[i][b];   // fractional dipoles change with the box
            }
        // reciprocal forces: -[q grad phi + (grad grad phi) . d]
        int K1 = s.grid[0], K2 = s.grid[1], K3 = s.grid[2], p = s.order;
        vector<V3> e = frac_dipoles(d);
        for (int i = 0; i < n; i++) {
            double f1[3] = {0, 0, 0}, f2[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
            for (int j1 = 0; j1 < p; j1++) {
                int g1 = ((gridIndex[0][i]-j1)%K1+K1)%K1;
                double t1 = theta[0][i*p+j1], u1 = dtheta[0][i*p+j1], v1 = d2theta[0][i*p+j1];
                for (int j2 = 0; j2 < p; j2++) {
                    int g2 = ((gridIndex[1][i]-j2)%K2+K2)%K2;
                    double t2 = theta[1][i*p+j2], u2 = dtheta[1][i*p+j2], v2 = d2theta[1][i*p+j2];
                    for (int j3 = 0; j3 < p; j3++) {
                        int g3 = ((gridIndex[2][i]-j3)%K3+K3)%K3;
                        double t3 = theta[2][i*p+j3], u3 = dtheta[2][i*p+j3], v3 = d2theta[2][i*p+j3];
                        double ph = phi[((size_t) g1*K2+g2)*K3+g3];
                        f1[0] += u1*t2*t3*ph;
                        f1[1] += t1*u2*t3*ph;
                        f1[2] += t1*t2*u3*ph;
                        f2[0][0] += v1*t2*t3*ph;
                        f2[1][1] += t1*v2*t3*ph;
                        f2[2][2] += t1*t2*v3*ph;
                        f2[0][1] += u1*u2*t3*ph;
                        f2[0][2] += u1*t2*u3*ph;
                        f2[1][2] += t1*u2*u3*ph;
                    }
                }
            }
            f2[1][0] = f2[0][1];
            f2[2][0] = f2[0][2];
            f2[2][1] = f2[1][2];
            for (int a = 0; a < 3; a++) {
                double c = q[i]*f1[a]+e[i][0]*f2[a][0]+e[i][1]*f2[a][1]+e[i][2]*f2[a][2];
                for (int k = 0; k < 3; k++)
                    F[i][k] -= s.grid[a]*recip[k][a]*c;
            }
        }
        double cs = 4.0*s.beta*s.beta*s.beta/(3.0*SQRT_PI);
        double q2 = 0, d2 = 0, Q = 0;
        for (int i = 0; i < n; i++) {
            q2 += q[i]*q[i];
            d2 += dot(d[i], d[i]);
            Q += q[i];
            for (int k = 0; k < 3; k++)
                E[i][k] += cs*d[i][k];
        }
        T.self = -(s.beta/SQRT_PI)*q2-0.5*cs*d2;
        T.background = -M_PI*Q*Q/(2.0*volume*s.beta*s.beta);
        for (int a = 0; a < 3; a++)
            S[4*a] -= T.background;   // E_bg ~ 1/V
    }
    for (int i = 0; i < n; i++)
        if (top.alpha[i] > 0)
            T.polarization += dot(mu[i], mu[i])/(2.0*top.alpha[i]);
    T.direct *= s.ke;
    T.reciprocal *= s.ke;
    T.self *= s.ke;
    T.background *= s.ke;
    T.polarization *= s.ke;
    lastTerms = T;
    if (forces) {
        // covalent dipoles: dE/dr_partner = c (g - (g.u) u)/|v|, g = dE/dd_atom = -KE E_atom
        for (size_t m = 0; m < top.covAtom.size(); m++) {
            int i = top.covAtom[m], j = top.covPartner[m];
            V3 v = {pos[j][0]-pos[i][0], pos[j][1]-pos[i][1], pos[j][2]-pos[i][2]};
            if (s.periodic)
                minImage(v, box);
            double L = sqrt(dot(v, v));
            V3 u = {v[0]/L, v[1]/L, v[2]/L};
            V3 g = {-E[i][0], -E[i][1], -E[i][2]};
            double gu = dot(g, u);
            for (int k = 0; k < 3; k++) {
                double t = top.covStrength[m]*(g[k]-gu*u[k])/L;
                F[j][k] -= t;
                F[i][k] += t;
                for (int b = 0; b < 3; b++)
                    S[3*k+b] += t*v[b];
            }
        }
        for (int k = 0; k < 9; k++)
            lastStrain[k] = s.ke*S[k];
        for (int i = 0; i < n; i++)
            for (int k = 0; k < 3; k++)
                (*forces)[i][k] += s.ke*F[i][k];
    }
    return T.total();
}

} // namespace pgmcore
