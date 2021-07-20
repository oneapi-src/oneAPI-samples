
#ifndef __DATA_BUNDLE_HPP__
#define __DATA_BUNDLE_HPP__

namespace hldutils {

template <typename T, int BUNDLE_SIZE>
struct DataBundle {
    T data_[BUNDLE_SIZE];

    DataBundle() {}

    DataBundle(const T op) {
        // unroll in case this constructor is implemented in an HLS component
#pragma unroll
        for (int idx = 0; idx < BUNDLE_SIZE; idx++) {
            data_[idx] = op;
        }
    }

    DataBundle(const DataBundle &op) {
        // unroll in case this copy constructor is implemented in an HLS component
#pragma unroll
        for (int idx = 0; idx < BUNDLE_SIZE; idx++) {
            data_[idx] = op.data_[idx];
        }
    }

    DataBundle& operator=(const DataBundle &op) {
        // unroll in case this copy constructor is implemented in an HLS component
#pragma unroll
        for (int idx = 0; idx < BUNDLE_SIZE; idx++) {
            data_[idx] = op.data_[idx];
        }
        return *this;
    }

    bool operator==(const DataBundle &rhs) {
        bool isEqual = true;
        // unroll in case this comparison is implemented in an HLS component
#pragma unroll
        for (int b = 0; b < BUNDLE_SIZE; b++) {
            isEqual &= (data_[b] == rhs.data_[b]);
        }

        return isEqual;
    }

    // get a specific value in the bundle
    T &operator[](int i) {
        return data_[i];
    }

    // get a raw pointer to underlying data
    T *data() {
        return &data_[0];
    }

    // For a shift register with N columns, the first piece of data is inserted in
    // index [N-1], and is read out of index [0].
    //
    // ```
    //         i=0  1   2
    //        ┌───┬───┬───┐
    // out ◄─ │ r ◄─e ◄─g ◄─ input
    //        └───┴───┴───┘
    // ```
    void shift(T &in) {
#pragma unroll
        for (int i = 0; i < (BUNDLE_SIZE - 1); i++) {
            data_[i] = data_[i + 1];
        }
        data_[BUNDLE_SIZE - 1] = in;
    }

    template <int SHIFT_AMT>
    void shiftSingleVal(T &in) {
#pragma unroll
        for (int i = 0; i < (BUNDLE_SIZE - SHIFT_AMT); i++) {
            data_[i] = data_[i + SHIFT_AMT];
        }

#pragma unroll
        for (int i = 0; i < (SHIFT_AMT); i++) {
            data_[(BUNDLE_SIZE - SHIFT_AMT) + i] = in;
        }
    }

    template <int SHIFT_AMT, int BUNDLE_SZ = SHIFT_AMT>
    void shiftMultiVals(DataBundle<T, BUNDLE_SZ> &in) {
#pragma unroll
        for (int i = 0; i < (BUNDLE_SIZE - SHIFT_AMT); i++) {
            data_[i] = data_[i + SHIFT_AMT];
        }

#pragma unroll
        for (int i = 0; i < (SHIFT_AMT); i++) {
            data_[(BUNDLE_SIZE - SHIFT_AMT) + i] = in[i];
        }
    }
};

} // namespace hldutils

#endif
