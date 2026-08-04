#include "StochasticPoissonSurfaceReconstruction.h"

SGP::JointKernelTerms<3> SGP::PSR3D::ComputeJointKernelTerms(const vec &x, const vec &p, scalar s)
{
    JointKernelTerms<3> t;
    t.s = s;

    vec y = x - p;
    scalar r = y.norm();
    t.r = r;

    scalar b = 1.0 / (std::sqrt(2.0) * s);

    if (r <= EPS_R) {
        // small-r expansion
        t.degenerate = true;
        scalar common = 1.0 / std::pow(PI, 1.5);
        scalar upp = (b*b*b) * (1.0/3.0) * common;

        t.grad_u = vec::Zero();
        t.hess_u = upp * mat::Identity();
        t.A = 0;
        t.Aprime = 0;
        return t;
    }

    t.degenerate = false;
    t.evec = y / r;

    // radial derivatives
    scalar br = b*r;
    scalar e = std::exp(-br*br);
    scalar E = std::erf(br);

    // u'(r)
    scalar up = - (1.0 / (4.0 * PI)) * ( (2.0*b/SQRT_PI)*e / r - E / (r*r) );
    // u''(r)
    scalar upp = ( (b*b*b)/std::pow(PI,1.5) ) * e + (b/std::pow(PI,1.5)) * e / (r*r) - E/(2.0*PI*r*r*r);
    // u'''(r)
    scalar uppp = -2.0*(b*b*b*b*b)/std::pow(PI,1.5) * r * e
                  -2.0*(b*b*b)/std::pow(PI,1.5) * e / r
                  -3.0*b/std::pow(PI,1.5) * e / (r*r*r)
                  + 3.0*E / (2.0 * PI * std::pow(r,4));

    t.grad_u = (up/r) * y;

    scalar B = up / r;
    t.A = upp - B;
    t.hess_u = t.A * (t.evec * t.evec.transpose()) + B * mat::Identity();
    t.Aprime = uppp - upp/r + up/(r*r);

    return t;
}

SGP::JointKernelTerms<2> SGP::PSR2D::ComputeJointKernelTerms(const vec2 &x, const vec2 &p, scalar s)
{
    JointKernelTerms<2> t;
    t.s = s;

    vec2 y = x - p;
    scalar r = y.norm();
    t.r = r;

    if (r <= EPS_R) {
        // small-r expansion
        t.degenerate = true;
        scalar val = 1.0 / (4.0 * PI * s * s);
        t.grad_u = val * y;
        t.hess_u = val * mat2::Identity();
        t.A = 0;
        t.Aprime = 0;
        return t;
    }

    t.degenerate = false;
    t.evec = y / r;

    // radial derivatives
    scalar z = (r*r)/(2.0*s*s);
    scalar ez = std::exp(-z);

    scalar up  = (1.0 - ez) / (2.0 * PI * r);
    scalar upp = (-1.0 + (1.0 + r*r/(s*s))*ez) / (2.0 * PI * r*r);
    scalar uppp = ((1.0 + r*r/(s*s)) * (-ez) * (r/(s*s)) - 2.0*upp)/r;

    t.grad_u = (up / r) * y;

    scalar B = up / r;
    t.A = upp - B;
    t.hess_u = t.A * (t.evec * t.evec.transpose()) + B * mat2::Identity();
    t.Aprime = uppp - upp / r + up / (r*r);

    return t;
}
