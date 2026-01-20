#include "StochasticPoissonSurfaceReconstruction.h"


SGP::JointPoissonKernel<3> SGP::PSR3D::ComputeJointPoissonKernel(const vec &x, const vec &p, const vec &n, scalar s)
{
    vec y = x - p;
    scalar r = y.norm();

    vec grad_u = vec::Zero();
    mat hess_u = mat::Zero();
    mat hess_of_dot = mat::Zero();

    scalar b = 1.0 / (std::sqrt(2.0) * s);
    if (r <= EPS_R) {
        // small-r expansion
        scalar common = 1.0 / std::pow(PI, 1.5);
        scalar up = (b*b*b) * r * (1.0/3.0) * common;
        scalar upp = (b*b*b) * (1.0/3.0) * common;
        scalar uppp = 0.0;

        grad_u = vec::Zero();//(up/r) * y;
        hess_u = upp * mat::Identity();
        hess_of_dot = mat::Zero();
    } else {
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

        // Gradient
        grad_u = (up/r) * y;

        // Hessian of u
        scalar B = up / r;
        scalar A = upp - B;
        mat outer = (y * y.transpose()) / (r*r);
        hess_u = A * outer + B * mat::Identity();

        // Hessian of ∇u·n
        scalar Aprime = uppp - upp/r + up/(r*r);
        vec evec = y / r;
        scalar edotn = evec.dot(n);
        mat eeT = evec * evec.transpose();
        mat I = mat::Identity();
        mat term1 = Aprime * edotn * eeT;
        mat term2 = (A/r) * edotn * (I - eeT);
        mat term3 = (A/r) * (evec * n.transpose() + n * evec.transpose());
        hess_of_dot = 0.5 * (term1 + term2 + term3 + (term1 + term2 + term3).transpose());
    }

    JointPoissonKernel<3> FK;

    PoissonKernel<3> Kp,Kn;
    Kp.row(0) = (hess_u*n).transpose();
    Kp.block(1,0,3,3) = hess_of_dot;

    Kn.row(0) = grad_u.transpose();
    Kn.block(1,0,3,3) = hess_u;

    FK.block(0,0,4,3) = Kp;
    FK.block(0,3,4,3) = Kn;


    FK *= s;

    return FK;
}

SGP::JointPoissonKernel<2> SGP::PSR2D::ComputeJointPoissonKernel(const vec2 &x, const vec2 &p, const vec2 &n, scalar s)
{
    vec2 y = x - p;
    scalar r = y.norm();

    vec2 grad_u = vec2::Zero();
    mat2 hess_u = mat2::Zero();
    mat2 hess_of_dot = mat2::Zero();

    if (r <= EPS_R) {
        // small-r expansion
        scalar val = 1.0 / (4.0 * PI * s * s);
        grad_u = val * y;
        hess_u = val * mat2::Identity();
        hess_of_dot.setZero();
    } else {

        // radial derivatives
        scalar z = (r*r)/(2.0*s*s);
        scalar ez = std::exp(-z);

        scalar up  = (1.0 - ez) / (2.0 * PI * r);
        scalar upp = (-1.0 + (1.0 + r*r/(s*s))*ez) / (2.0 * PI * r*r);
        scalar uppp = ((1.0 + r*r/(s*s)) * (-ez) * (r/(s*s)) - 2.0*upp)/r; // u'''(r) simplified

        // gradient
        grad_u = (up / r) * y;

        // Hessian of u
        scalar B = up / r;
        scalar A = upp - B;
        mat2 outer = (y * y.transpose()) / (r*r);
        hess_u = A * outer + B * mat2::Identity();

        // Hessian of ∇u·n
        scalar Aprime = uppp - upp / r + up / (r*r);
        vec2 evec = y / r;
        scalar edotn = evec.dot(n);
        mat2 eeT = evec * evec.transpose();
        mat2 I = mat2::Identity();
        mat2 term1 = Aprime * edotn * eeT;
        mat2 term2 = (A / r) * edotn * (I - eeT);
        mat2 term3 = (A / r) * (evec * n.transpose() + n * evec.transpose());
        mat2 H = term1 + term2 + term3;
        hess_of_dot = 0.5 * (H + H.transpose()); // ensure symmetry
    }

    JointPoissonKernel<2> FK;

    PoissonKernel<2> Kp,Kn;
    Kp.row(0) = (hess_u*n).transpose();
    Kp.block(1,0,2,2) = hess_of_dot;

    Kn.row(0) = grad_u.transpose();
    Kn.block(1,0,2,2) = hess_u;

    FK.block(0,0,3,2) = Kp;
    FK.block(0,2,3,2) = Kn;


    FK *= s;

    return FK;
}
