#ifndef TENSOR_H
#define TENSOR_H

#include "types.h"

namespace SGP {

template<class T>
inline T Zero() {
    return T();
}

template<>
inline vec2 Zero<vec2>() {return vec2::Zero();}

template<>
inline vec Zero<vec>() {return vec::Zero();}

template<int N_DIM, class T>
class Tensor {
public:
    using SliceIndex = Eigen::Vector<Index,N_DIM>;
private:
    using Values = Eigen::Vector<T,-1>;
    Values values;
    SliceIndex dims;
public:


    Index getSize() const {
        Index size = 1;
        for (int i = 0; i < N_DIM; i++) {
            size *= dims[i];
        }
        return size;
    }

    const SliceIndex& getSizes() const {
        return dims;
    }

    //variadic function to set sizes
    template<typename... Args>
    void setSizes(Args... args) {
        Index sizes[] = {args...};
        for (int i = 0; i < N_DIM; i++) {
            dims[i] = sizes[i];
        }
        values = Values::Constant(getSize(),Zero<T>());
    }

    Tensor(const SliceIndex& S) :dims(S) {
        Index size = 1;
        for (int i = 0; i < N_DIM; i++)
            size *= dims[i];
        values = Values::Constant(size, Zero<T>());
    }

    Tensor() {}


    //variadic constructor to set sizes
    template<typename... Args>
    Tensor(Args... args) {
        setSizes(args...);
    }

    // variadic function to get index
    template<typename... Args>
    Index getIndex(Args... args) const {
        Index index = 0;
        Index mult = 1;
        Index indices[] = {args...};
        for (int i = 0; i < N_DIM; i++) {
            index += indices[i] * mult;
            mult *= dims[i];
        }
        return index;
    }

    SliceIndex getSliceIndexes(Index index) const {
        SliceIndex slice;
        Index mult = 1;
        for (int i = 0; i < N_DIM; i++) {
            slice[i] = (index / mult) % dims[i];
            mult *= dims[i];
        }
        return slice;
    }

    Index getIndex(const SliceIndex& index) const {
        Index i = 0;
        Index mult = 1;
        for (int j = 0; j < N_DIM; j++) {
            i += index[j] * mult;
            mult *= dims[j];
        }
        return i;
    }

    void fill(const T& value) {
        values.fill(value);
    }

    T& at(Index index) {
        return values(index);
    }

    const T& at(Index index) const {
        return values(index);
    }

    T& operator()(const SliceIndex& index) {
        return values(getIndex(index));
    }

    template<typename... Args>
    T& operator()(Args... args) {
        return values(getIndex(args...));
    }

    template<typename... Args>
    const T& operator()(Args... args) const {
        return values(getIndex(args...));
    }

    const T& operator()(const SliceIndex& index) const {
        return values(getIndex(index));
    }

    Values& data() {
        return values;
    }

    const Values& data() const {
        return values;
    }

    template<class out,class func>
    Tensor<N_DIM,out> apply(const func& f) const {
        Tensor<N_DIM,out> rslt(dims);
        #pragma omp parallel for
        for (Index i = 0; i < getSize(); i++)
            rslt.at(i) = f(at(i));
        return rslt;
    }




    bool validIndex(const SliceIndex& index) const {
        for (int i = 0; i < N_DIM; i++) {
            if (index[i] < 0 || index[i] >= dims[i]) {
                return false;
            }
        }
        return true;
    }

};

}

#endif // TENSOR_H
