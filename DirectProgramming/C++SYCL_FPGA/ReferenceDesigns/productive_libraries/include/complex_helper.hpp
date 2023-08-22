#pragma once
#include <array>
#include <complex>
#include <sycl/sycl.hpp>
#include <utility>

namespace t2sp {
namespace detail {

template <typename T>
struct unwrap {};
template <typename T>
struct unwrap<std::complex<T>> {
    using type = T;
};

template <std::size_t... Is, typename F>
constexpr void loop(std::index_sequence<Is...>, F &&f) {
    (f(std::integral_constant<std::size_t, Is>{}), ...);
}

template <std::size_t N, typename F>
constexpr void loop(F &&f) {
    loop(std::make_index_sequence<N>{}, std::forward<F&&>(f));
}

template <typename T, size_t N>
class vec {
    using unwrap_t = typename unwrap<T>::type;
    sycl::vec<unwrap_t, 2 * N> _data;
    template <typename A, size_t... Is>
    void make_vec(const std::array<A, N> &arr, std::index_sequence<Is...>) {
        ((_data[2 * Is] = arr[Is].real()), ...);
        ((_data[2 * Is + 1] = arr[Is].imag()), ...);
    }

  public:
    vec() = default;
    vec(const vec &) = default;
    vec(vec &&) = default;
    vec &operator=(const vec &) = default;
    vec &operator=(vec &&) = default;
    vec(const T &arg) {
        loop<N>([&](auto n) {
            constexpr size_t i = decltype(n)::value;
            _data[2 * i] = arg.real();
            _data[2 * i + 1] = arg.imag();
        });
    }
    template <typename A, std::enable_if_t<std::is_arithmetic_v<A>, int> = 0>
    vec(A arg) : vec(T{static_cast<unwrap_t>(arg)}) {}
    template <typename... Args>
    vec(const Args &...args) {
        make_vec(std::array{args...}, std::make_index_sequence<N>());
    }
    template <typename A>
    vec &operator=(const A &arg) {
        loop<N>([&](auto n) {
            constexpr size_t i = decltype(n)::value;
            _data[2 * i] = arg.real();
            _data[2 * i + 1] = arg.imag();
        });
        return *this;
    }
    vec<T, 1> &operator[](size_t n) { return reinterpret_cast<vec<T, 1> *>(&_data)[n]; }
    const vec<T, 1> &operator[](size_t n) const {
        return reinterpret_cast<const vec<T, 1> *>(&_data)[n];
    }
    constexpr static size_t size() noexcept { return N; }
    vec &operator+=(const vec &rhs) {
        _data += rhs._data;
        return *this;
    }
    template <typename A>
    vec &operator+=(const A &rhs) {
        loop<N>([&](auto n) {
            constexpr size_t i = decltype(n)::value;
            _data[2 * i] += rhs.real();
            _data[2 * i + 1] += rhs.imag();
        });
        return *this;
    }
    vec &operator-=(const vec &rhs) {
        _data -= rhs._data;
        return *this;
    }
    vec &operator-=(const T &rhs) {
        loop<N>([&](auto n) {
            constexpr size_t i = decltype(n)::value;
            _data[2 * i] -= rhs.real();
            _data[2 * i + 1] -= rhs.imag();
        });
        return *this;
    }
    vec &operator*=(const vec &rhs) {
        loop<N>([&](auto n) {
            constexpr size_t i = decltype(n)::value;
            _data[2 * i] = _data[2 * i] * rhs._data[2 * i] -
                           _data[2 * i + 1] * rhs._data[2 * i + 1];
            _data[2 * i + 1] = _data[2 * i] * rhs._data[2 * i + 1] +
                               _data[2 * i + 1] * rhs._data[2 * i];
        });
        return *this;
    }
    template <typename A>
    vec &operator*=(const A &arg) {
        loop<N>([&](auto n) {
            constexpr size_t i = decltype(n)::value;
            _data[2 * i] =
                _data[2 * i] * arg.real() - _data[2 * i + 1] * arg.imag();
            _data[2 * i + 1] =
                _data[2 * i] * arg.imag() + _data[2 * i + 1] * arg.real();
        });
        return *this;
    }
    friend vec operator+(const vec &arg) { return arg; }
    friend vec operator-(const vec &arg) {
        vec ret{};
        ret._data = -arg._data;
        return ret;
    }
    friend vec operator+(const vec &lhs, const vec &rhs) {
        vec ret{};
        ret._data = lhs._data + rhs._data;
        return ret;
    }
    template <typename A>
    friend vec operator+(const vec &lhs, const A &rhs) {
        vec ret{};
        loop<N>([&](auto n) {
            constexpr size_t i = decltype(n)::value;
            ret._data[2 * i] = lhs._data[2 * i] + rhs.real();
            ret._data[2 * i + 1] = lhs._data[2 * i + 1] + rhs.imag();
        });
        return ret;
    }
    template <typename A>
    friend vec operator+(const A &lhs, const vec &rhs) {
        vec ret{};
        loop<N>([&](auto n) {
            constexpr size_t i = decltype(n)::value;
            ret._data[2 * i] = lhs.real() + rhs._data[2 * i];
            ret._data[2 * i + 1] = lhs.imag() + rhs._data[2 * i + 1];
        });
        return ret;
    }
    friend vec operator-(const vec &lhs, const vec &rhs) {
        vec ret{};
        ret._data = lhs._data - rhs._data;
        return ret;
    }
    template <typename A>
    friend vec operator-(const vec &lhs, const A &rhs) {
        vec ret{};
        loop<N>([&](auto n) {
            constexpr size_t i = decltype(n)::value;
            ret._data[2 * i] = lhs._data[2 * i] - rhs.real();
            ret._data[2 * i + 1] = lhs._data[2 * i + 1] - rhs.imag();
        });
        return ret;
    }
    template <typename A>
    friend vec operator-(const A &lhs, const vec &rhs) {
        vec ret{};
        loop<N>([&](auto n) {
            constexpr size_t i = decltype(n)::value;
            ret._data[2 * i] = lhs.real() - rhs._data[2 * i];
            ret._data[2 * i + 1] = lhs.imag() - rhs._data[2 * i + 1];
        });
        return ret;
    }
    friend vec operator*(const vec &lhs, const vec &rhs) {
        vec ret{};
        loop<N>([&](auto n) {
            constexpr size_t i = decltype(n)::value;
            ret._data[2 * i] = lhs._data[2 * i] * rhs._data[2 * i] -
                               lhs._data[2 * i + 1] * rhs._data[2 * i + 1];
            ret._data[2 * i + 1] = lhs._data[2 * i] * rhs._data[2 * i + 1] +
                                   lhs._data[2 * i + 1] * rhs._data[2 * i];
        });
        return ret;
    }
    template <typename A>
    friend vec operator*(const vec &lhs, const A &rhs) {
        vec ret{};
        loop<N>([&](auto n) {
            constexpr size_t i = decltype(n)::value;
            ret._data[2 * i] = lhs._data[2 * i] * rhs.real() -
                               lhs._data[2 * i + 1] * rhs.imag();
            ret._data[2 * i + 1] = lhs._data[2 * i] * rhs.imag() +
                                   lhs._data[2 * i + 1] * rhs.real();
        });
        return ret;
    }
    template <typename A>
    friend vec operator*(const A &lhs, const vec &rhs) {
        vec ret{};
        loop<N>([&](auto n) {
            constexpr size_t i = decltype(n)::value;
            ret._data[2 * i] = lhs.real() * rhs._data[2 * i] -
                               lhs.imag() * rhs._data[2 * i + 1];
            ret._data[2 * i + 1] = lhs.real() * rhs._data[2 * i + 1] +
                                   lhs.imag() * rhs._data[2 * i];
        });
        return ret;
    }
    friend vec operator*(const vec &lhs, const unwrap_t &rhs) {
        vec ret{};
        ret._data = lhs._data * rhs;
        return ret;
    }
    friend vec operator*(const unwrap_t &lhs, const vec &rhs) {
        vec ret{};
        ret._data = lhs * rhs._data;
        return ret;
    }
    friend bool operator==(const vec &lhs, const vec &rhs) {
        return sycl::all(lhs._data == rhs._data);
    }
    template <typename A>
    friend bool operator==(const vec &lhs, const A &rhs) {
        bool ret = true;
        loop<N>([&](auto n) {
            constexpr size_t i = decltype(n)::value;
            ret = ret && lhs._data[2 * i] == rhs.real();
            ret = ret && lhs._data[2 * i + 1] == rhs.imag();
        });
        return ret;
    }
    template <typename A>
    friend bool operator==(const A &lhs, const vec &rhs) {
        bool ret = true;
        loop<N>([&](auto n) {
            constexpr size_t i = decltype(n)::value;
            ret = ret && lhs.real() == rhs._data[2 * i];
            ret = ret && lhs.imag() == rhs._data[2 * i + 1];
        });
        return ret;
    }
    friend bool operator!=(const vec &lhs, const vec &rhs) {
        return !(lhs == rhs);
    }
    template <typename A>
    friend bool operator!=(const vec &lhs, const A &rhs) {
        return !(lhs == rhs);
    }
    template <typename A>
    friend bool operator!=(const A &lhs, const vec &rhs) {
        return !(lhs == rhs);
    }
};

template <typename T>
class vec<T, 1> {
    using unwrap_t = typename unwrap<T>::type;
    sycl::vec<unwrap_t, 2> _data;

  public:
    vec() = default;
    vec(const vec &) = default;
    vec(vec &&) = default;
    vec &operator=(const vec &) = default;
    vec &operator=(vec &&) = default;
    vec(const T &arg) {
        _data[0] = arg.real();
        _data[1] = arg.imag();
    }
    template <typename A, std::enable_if_t<std::is_arithmetic_v<A>, int> = 0>
    vec(A arg) {
        _data[0] = arg;
        _data[1] = 0;
    }
    template <typename A, std::enable_if_t<std::is_arithmetic_v<A>, int> = 0>
    vec(A arg0, A arg1) {
        _data[0] = arg0;
        _data[1] = arg1;
    }
    auto real() const {
        return _data[0];
    }
    auto imag() const {
        return _data[1];
    }
    operator T() const {
        return T{_data[0], _data[1]};
    }
    vec &operator+=(const vec &rhs) {
        _data += rhs._data;
        return *this;
    }
    vec &operator+=(const T &rhs) {
        _data[0] += rhs.real();
        _data[1] += rhs.imag();
        return *this;
    }
    vec &operator-=(const vec &rhs) {
        _data -= rhs._data;
        return *this;
    }
    vec &operator-=(const T &rhs) {
        _data[0] -= rhs.real();
        _data[1] -= rhs.imag();
        return *this;
    }
    template <typename A>
    vec &operator*=(const A &arg) {
        _data[0] =
            _data[0] * arg.real() - _data[1] * arg.imag();
        _data[1] =
            _data[0] * arg.imag() + _data[1] * arg.real();
        return *this;
    }
    vec conj() const {
        vec ret{};
        ret._data[0] = _data[0];
        ret._data[1] = -_data[1];
        return ret;
    }
    vec sqrt() const {
        vec ret{};
        const auto rho = sycl::sqrt(sycl::sqrt(_data[0] * _data[0] + _data[1] * _data[1]));
        const auto theta = sycl::atan2(_data[1], _data[0]) / 2;
        ret._data[0] = rho * sycl::cos(theta);
        ret._data[1] = rho * sycl::sin(theta);
        return ret;
    }
    friend vec operator+(const vec &arg) { return arg; }
    friend vec operator-(const vec &arg) {
        vec ret{};
        ret._data = -arg._data;
        return ret;
    }
    friend vec operator+(const vec &lhs, const vec &rhs) {
        vec ret{};
        ret._data = lhs._data + rhs._data;
        return ret;
    }
    friend vec operator-(const vec &lhs, const vec &rhs) {
        vec ret{};
        ret._data = lhs._data - rhs._data;
        return ret;
    }
    friend vec operator*(const vec &lhs, const vec &rhs) {
        vec ret{};
        ret._data[0] = lhs._data[0] * rhs._data[0] -
                       lhs._data[1] * rhs._data[1];
        ret._data[1] = lhs._data[0] * rhs._data[1] +
                       lhs._data[1] * rhs._data[0];
        return ret;
    }
    friend vec operator*(const unwrap_t &lhs, const vec &rhs) {
        vec ret{};
        ret._data = lhs * rhs._data;
        return ret;
    }
    friend vec operator*(const vec &lhs, const unwrap_t &rhs) {
        return rhs * lhs;
    }
    friend bool operator==(const vec &lhs, const vec &rhs) {
        return sycl::all(lhs._data == rhs._data);
    }
    friend bool operator!=(const vec &lhs, const vec &rhs) {
        return !(lhs == rhs);
    }
#define define_op(op) \
    friend vec operator op(const vec &lhs, const T &rhs) {\
        return lhs op vec{rhs};\
    }\
    friend vec operator op(const T &lhs, const vec &rhs) {\
        return vec{lhs} op rhs;\
    }

    define_op(+)
    define_op(-)
    define_op(*)
    define_op(==)
    define_op(!=)
};
}  // namespace detail
}  // namespace t2sp

namespace std {
template <typename T>
auto sqrt(const t2sp::detail::vec<std::complex<T>, 1> &r) {
    return r.sqrt();
}
template <typename T>
auto conj(const t2sp::detail::vec<std::complex<T>, 1> &r) {
    return r.conj();
}
template <typename T>
struct is_trivially_copyable<t2sp::detail::vec<std::complex<T>, 1>> {
    constexpr static auto value = true;
};
}  // namespace std

using complexf = t2sp::detail::vec<std::complex<float>, 1>;
using complexf2 = t2sp::detail::vec<std::complex<float>, 2>;
using complexf4 = t2sp::detail::vec<std::complex<float>, 4>;
using complexf8 = t2sp::detail::vec<std::complex<float>, 8>;
using complexd = t2sp::detail::vec<std::complex<double>, 1>;
using complexd2 = t2sp::detail::vec<std::complex<double>, 2>;
using complexd4 = t2sp::detail::vec<std::complex<double>, 4>;
using complexd8 = t2sp::detail::vec<std::complex<double>, 8>;

