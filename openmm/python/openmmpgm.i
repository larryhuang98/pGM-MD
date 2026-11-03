%module openmmpgm

%import(module="openmm") "swig/OpenMMSwigHeaders.i"
%include "swig/typemaps.i"
%include "std_vector.i"
namespace std {
  %template(vectord) vector<double>;
  %template(vectori) vector<int>;
};

%{
#include "GVDWForce.h"
#include "PGMForce.h"
#include "OpenMM.h"
#include "OpenMMAmoeba.h"
#include "OpenMMDrude.h"
#include "openmm/RPMDIntegrator.h"
#include "openmm/RPMDMonteCarloBarostat.h"
%}

%pythoncode %{
import openmm as mm
import openmm.unit as unit
%}

%exception {
    try {
        $action
    } catch (std::exception &e) {
        PyErr_SetString(PyExc_Exception, const_cast<char*>(e.what()));
        return NULL;
    }
}

namespace PGMPlugin {

class PGMForce : public OpenMM::Force {
public:
    enum NonbondedMethod { NoCutoff = 0, PME = 1 };
    enum Polarization { Mutual = 0, NoPolarization = 1 };
    enum Predictor { NoPredictor = 0, Mu3 = 1, Mu4 = 2 };
    PGMForce();
    int getNumParticles() const;
    int addParticle(double charge, double radius, double polarizability);
    %apply double& OUTPUT {double& charge};
    %apply double& OUTPUT {double& radius};
    %apply double& OUTPUT {double& polarizability};
    void getParticleParameters(int index, double& charge, double& radius, double& polarizability) const;
    %clear double& charge;
    %clear double& radius;
    %clear double& polarizability;
    void setParticleParameters(int index, double charge, double radius, double polarizability);
    int getNumCovalentDipoles() const;
    int addCovalentDipole(int particle, int partner, double strength);
    %apply int& OUTPUT {int& particle};
    %apply int& OUTPUT {int& partner};
    %apply double& OUTPUT {double& strength};
    void getCovalentDipoleParameters(int index, int& particle, int& partner, double& strength) const;
    %clear int& particle;
    %clear int& partner;
    %clear double& strength;
    void setCovalentDipoleParameters(int index, int particle, int partner, double strength);
    NonbondedMethod getNonbondedMethod() const;
    void setNonbondedMethod(NonbondedMethod method);
    double getCutoffDistance() const;
    void setCutoffDistance(double distance);
    %apply double& OUTPUT {double& alpha};
    %apply int& OUTPUT {int& nx};
    %apply int& OUTPUT {int& ny};
    %apply int& OUTPUT {int& nz};
    void getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const;
    void getPMEParametersInContext(const OpenMM::Context& context, double& alpha, int& nx, int& ny, int& nz) const;
    %clear double& alpha;
    %clear int& nx;
    %clear int& ny;
    %clear int& nz;
    void setPMEParameters(double alpha, int nx, int ny, int nz);
    int getPMEOrder() const;
    void setPMEOrder(int order);
    double getEwaldErrorTolerance() const;
    void setEwaldErrorTolerance(double tol);
    Polarization getPolarization() const;
    void setPolarization(Polarization type);
    double getInducedDipoleTolerance() const;
    void setInducedDipoleTolerance(double tol);
    int getMaxInducedDipoleIterations() const;
    void setMaxInducedDipoleIterations(int iter);
    double getPeekCoefficient() const;
    void setPeekCoefficient(double omega);
    Predictor getPredictor() const;
    void setPredictor(Predictor predictor);
    double getCoulombConstant() const;
    void setCoulombConstant(double ke);
    int getInducedDipoleIterations(OpenMM::Context& context);
    void updateParametersInContext(OpenMM::Context& context);
    bool usesPeriodicBoundaryConditions() const;

    %extend {
        /* dipoles as flat lists [x0, y0, z0, x1, ...] in e nm (wrapped below as N x 3 numpy arrays) */
        std::vector<double> _inducedDipoles(OpenMM::Context& context) {
            std::vector<OpenMM::Vec3> d;
            self->getInducedDipoles(context, d);
            std::vector<double> out(3*d.size());
            for (size_t i = 0; i < d.size(); i++)
                for (int k = 0; k < 3; k++)
                    out[3*i+k] = d[i][k];
            return out;
        }
        std::vector<double> _permanentDipoles(OpenMM::Context& context) {
            std::vector<OpenMM::Vec3> d;
            self->getPermanentDipoles(context, d);
            std::vector<double> out(3*d.size());
            for (size_t i = 0; i < d.size(); i++)
                for (int k = 0; k < 3; k++)
                    out[3*i+k] = d[i][k];
            return out;
        }
        static PGMPlugin::PGMForce& cast(OpenMM::Force& force) {
            return dynamic_cast<PGMPlugin::PGMForce&>(force);
        }
        static bool isinstance(OpenMM::Force& force) {
            return (dynamic_cast<PGMPlugin::PGMForce*>(&force) != NULL);
        }
        %pythoncode %{
          def getInducedDipoles(self, context):
              """Induced dipoles of the last evaluation, (N, 3) numpy array [e nm]."""
              import numpy as np
              return np.array(self._inducedDipoles(context)).reshape(-1, 3)

          def getPermanentDipoles(self, context):
              """Covalent permanent dipoles at the current positions, (N, 3) numpy array [e nm]."""
              import numpy as np
              return np.array(self._permanentDipoles(context)).reshape(-1, 3)

          def getTotalDipoles(self, context):
              return self.getPermanentDipoles(context)+self.getInducedDipoles(context)
        %}
    }
};

class GVDWForce : public OpenMM::Force {
public:
    enum NonbondedMethod { NoCutoff = 0, CutoffNonPeriodic = 1, CutoffPeriodic = 2 };
    enum RepulsionForm { Gaussian = 0, Slater = 1 };
    GVDWForce();
    int getNumParticles() const;
    int addParticle(double radius, double sqrtA, double sqrtC6, double b);
    %apply double& OUTPUT {double& radius};
    %apply double& OUTPUT {double& sqrtA};
    %apply double& OUTPUT {double& sqrtC6};
    %apply double& OUTPUT {double& b};
    void getParticleParameters(int index, double& radius, double& sqrtA, double& sqrtC6, double& b) const;
    %clear double& radius;
    %clear double& sqrtA;
    %clear double& sqrtC6;
    %clear double& b;
    void setParticleParameters(int index, double radius, double sqrtA, double sqrtC6, double b);
    int getNumExclusions() const;
    int addExclusion(int particle1, int particle2);
    %apply int& OUTPUT {int& particle1};
    %apply int& OUTPUT {int& particle2};
    void getExclusionParticles(int index, int& particle1, int& particle2) const;
    %clear int& particle1;
    %clear int& particle2;
    void setExclusionParticles(int index, int particle1, int particle2);
    NonbondedMethod getNonbondedMethod() const;
    void setNonbondedMethod(NonbondedMethod method);
    double getCutoffDistance() const;
    void setCutoffDistance(double distance);
    RepulsionForm getRepulsionForm() const;
    void setRepulsionForm(RepulsionForm form);
    bool getUseDispersionCorrection() const;
    void setUseDispersionCorrection(bool use);
    void updateParametersInContext(OpenMM::Context& context);
    bool usesPeriodicBoundaryConditions() const;
    %extend {
        static PGMPlugin::GVDWForce& cast(OpenMM::Force& force) {
            return dynamic_cast<PGMPlugin::GVDWForce&>(force);
        }
        static bool isinstance(OpenMM::Force& force) {
            return (dynamic_cast<PGMPlugin::GVDWForce*>(&force) != NULL);
        }
    }
};

}
