#pragma once

#include "types.h"
#include <random>

namespace SGP {

struct PCG32
{
    PCG32( ) : x(), key() { seed(0x853c49e6748fea9b, c); }
    PCG32( const uint64_t s, const uint64_t ss= c ) : x(), key() { seed(s, ss); }

    void seed( const uint64_t s, const uint64_t ss= c )
    {
        key= (ss << 1) | 1;

        x= key + s;
        sample();
    }

    unsigned sample( )
    {
        // f(x), fonction de transition
        uint64_t xx= x;
        x= a*x + key;

        // g(x), fonction résultat
        uint32_t tmp= ((xx >> 18u) ^ xx) >> 27u;
        uint32_t r= xx >> 59u;
        return (tmp >> r) | (tmp << ((~r + 1u) & 31));
    }

    // c++ interface
    unsigned operator() ( ) { return sample(); }
    static constexpr unsigned min( ) { return 0; }
    static constexpr unsigned max( ) { return ~unsigned(0); }
    typedef unsigned result_type;

    static constexpr uint64_t a= 0x5851f42d4c957f2d;
    static constexpr uint64_t c= 0xda3e39cb94b95bdb;

    uint64_t x;
    uint64_t key;
};



inline Vecs sampleUnitGaussian(int N,int dim) {
    //static std::mt19937 gen;

    static std::random_device hwseed;
    static PCG32 gen( hwseed(), hwseed() );
    static std::normal_distribution<scalar> dist{0.0,1.0};
    Vecs X(N,Vec(dim));
    for (auto& x : X){
        for (int i = 0;i<dim;i++)
            x(i) = dist(gen);
    }
    return X;
}


inline Vec sampleUnitGaussian(int dim) {
    /*
    static thread_local std::random_device hwseed;
    static thread_local PCG32 rng( hwseed(), hwseed() );
*/
    std::normal_distribution<scalar> dist{0.0,1.0};
    //static thread_local std::mt19937 gen;
    static thread_local std::random_device rd;
    static thread_local std::mt19937 rng(rd());
    Vec X(dim);
    for (int i = 0;i<dim;i++)
        X(i) = dist(rng);
    return X;
}

inline Mat sampleUnitGaussianMat(int n,int dim) {
    static thread_local std::random_device rd;
    static thread_local std::mt19937 gen(rd());
    std::normal_distribution<scalar> dist{0.0,1.0};
    Mat X(dim,n);
    for (auto i : range(dim))
        for (auto j : range(n))
            X(i,j) = dist(gen);
    return X;
}

inline Mat sampleUnitSphere(int n,int dim) {
    static std::mt19937 gen;
    static std::normal_distribution<scalar> dist{0.0,1.0};
    Mat X(dim,n);
    for (auto i : range(n)){
        for (auto j : range(dim))
            X(j,i) = dist(gen);
        X.col(i).normalize();
    }
    return X;
}

inline Mat sampleUnitSquare(int n,int dim) {
    static std::mt19937 gen;
    static std::normal_distribution<scalar> dist{0.0,1.0};
    Mat X(dim,n);
    for (auto i : range(n)){
        for (auto j : range(dim))
            X(j,i) = dist(gen);
        X.col(i) /= X.col(i).lpNorm<Eigen::Infinity>();
    }
    return X;
}


template<class T>
size_t WeightedRandomChoice(const T& weights) {
    // Random number generator
    static std::random_device rd;
    static std::mt19937 gen(rd());

    // Create a discrete distribution based on the weights
    std::discrete_distribution<> dist(weights.begin(), weights.end());

    // Draw an index based on weights
    return dist(gen);
}

inline Vecs fibonacci_sphere(int n)
{
    static double goldenRatio = (1 + std::sqrt(5.))/2.;
    Vecs FS(n);
    for (int i = 0;i<n;i++){
        double theta = 2 * M_PI * i / goldenRatio;
        double phi = std::acos(1 - 2*(i+0.5)/n);
        FS[i] = vec(cos(theta) * sin(phi), sin(theta) * sin(phi), cos(phi));
    }
    return FS;
}

inline Vecs Uniform(int n,int d)
{
    Vecs U(n);
    for (int i = 0;i<n;i++)
        U[i] = Vec::Random(d);
    return U;
}


// Fonction pour générer un point uniforme dans la boule unité en dimension d
inline Vec sample_point_in_unit_ball(int d) {
    static std::mt19937 gen;
    static std::normal_distribution<double> gaussian_dist;
    static std::uniform_real_distribution<double> uniform_dist;
    // Génère un point gaussien aléatoire
    Vec point(d);
    for (int i = 0; i < d; ++i) {
        point[i] = gaussian_dist(gen);
    }

    // Normalisation pour obtenir un point sur la sphère
    point.normalize();

    // Distance aléatoire à l'intérieur de la boule avec distribution uniforme
    double radius = std::pow(uniform_dist(gen), 1.0 / d);

    return point * radius;
}

// Fonction principale pour échantillonner N points dans la boule unité de dimension d
inline Vecs sample_unit_ball(int N, int d,double r = 1,Vec offset = Vec()) {
    Vecs samples(N);
    if (!offset.size())
        offset = Vec::Zero(d);

    for (int i = 0; i < N; ++i)
        samples[i] = sample_point_in_unit_ball(d)*r + offset;

    return samples;
}

inline Vecs sampleGaussian(int dim,int N,const Vec& mean,const Mat& Cov) {
    Vecs X = sampleUnitGaussian(N,dim);
    for (auto& x : X)
        x = Cov*x + mean;
    return X;
}

inline Mat sampleUnitBall(int N,int d) {
    static std::mt19937 gen;
    static std::normal_distribution<double> gaussian_dist;
    static std::uniform_real_distribution<double> uniform_dist;

    Mat X(d,N);
    for (auto i : range(N)){
        Vec point(d);
        for (int j = 0; j < d; ++j) {
            point[j] = gaussian_dist(gen);
        }

        // Normalisation pour obtenir un point sur la sphère
        point.normalize();

        // Distance aléatoire à l'intérieur de la boule avec distribution uniforme
        double radius = std::pow(uniform_dist(gen), 1.0 / d);

        X.col(i) = point * radius;
    }
    return X;
}

}
