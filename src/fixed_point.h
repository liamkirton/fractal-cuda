//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<uint32_t I, uint32_t F>
class fixed_point {
public:
    __host__ __device__ fixed_point() : data{ 0 } {
    }

    __host__ __device__ fixed_point(const fixed_point<I, F> &v) {
        memcpy(&data, &v.data, sizeof(data));
    }

    __host__ __device__ fixed_point(const fixed_point<I, F> *v) {
        memcpy(&data, &v->data, sizeof(data));
    }

    template<typename T>
    __host__ __device__ fixed_point(const T &v) {
        set(v);
    }

    __host__ __device__ fixed_point(const char *v) {
        set(std::string(v));
    }

    //
    // Accessors & Initialisers
    //

    inline __host__ __device__ operator double() {
        return get_double();
    }

    inline __host__ __device__ uint32_t bit_get(const uint64_t i) const {
        if (i < (I + F) * 32) {
            return (data[i / 32] & (1 << (i % 32))) >> (i % 32);
        }
        return 0;
    }

    inline __host__ __device__ void bit_set(const uint64_t i, const uint64_t v) {
        if (i < (I + F) * 32) {
            data[i / 32] = data[i / 32] | ((v & 0x1) << (i % 32));
        }
    }

    inline __host__ __device__ double get_double() const {
        fixed_point<I, F> t(*this);
        if (t.bit_get((I + F) * 32 - 1) == 1) {
            t.negate();
        }
        double value = 1.0;
        int64_t integer = t.get_integer();
        uint64_t fractional = 0;
        if (F > 1) {
            fractional = *reinterpret_cast<const uint64_t *>(&data[F - 2]);
        }
        else {
            fractional = *reinterpret_cast<const uint32_t *>(&data[F - 1]);
        }
        *reinterpret_cast<uint64_t *>(&value) |= (fractional >> 12);
        value -= 1.0;
        value += 1.0 * integer;
        if (get_integer() < 0) {
            value *= -1.0;
        }
        return value;
    }

    inline __host__ __device__ int64_t get_integer() const {
        if (I > 1) {
            return *reinterpret_cast<const int64_t *>(&data[F]);
        }
        return static_cast<const int64_t>(*reinterpret_cast<const int32_t *>(&data[F]));
    }

    inline __host__ __device__ void set(const fixed_point<I, F> &v) {
        memcpy(&data, &v.data, sizeof(data));
    }

    template<uint32_t I_X>
    inline __host__ __device__ void set(const fixed_point<I_X, F> &v) {
        memcpy(&data, &v.data, sizeof(uint32_t) * (((I_X < I) ? I_X : I) + F));
        if (I_X < I) {
            memset(&data[I_X + F], (v.bit_get(32 * (I_X + F) - 1) == 1) ? 0xff : 0, sizeof(uint32_t) * (I - I_X));
        }
    }

    inline __host__ __device__ void set(const int &v) {
        set(static_cast<int64_t>(v));
    }

    inline __host__ __device__ void set(const unsigned int &v) {
        set(static_cast<uint64_t>(v));
    }

    inline __host__ __device__ void set(const int64_t &v) {
        zero();
        data[F] = v & 0xffffffff;
        if (2 <= I) {
            data[F + 1] = (v >> 32);
        }
        if ((2 < I) && (v & 0x8000000000000000)) {
            for (uint32_t i = F + 2; i < I + F; ++i) {
                data[i] = 0xffffffff;
            }
        }
    }

    inline __host__ __device__ void set(const uint64_t &v) {
        zero();
        data[F] = v & 0xffffffff;
        if (2 <= I) {
            data[F + 1] = (v >> 32);
        }
    }

    inline __host__ __device__ void set(const double &v) {
        double integer{ 0 };
        double decimal = modf(fabs(v), &integer);

        fixed_point<I, F> integer_part(static_cast<int64_t>(integer));

        fixed_point<I, F> decimal_part;
        for (uint32_t i = 0; i < 32 * F; ++i) {
            decimal *= 2;
            decimal = modf(decimal, &integer);
            decimal_part.bit_set(32 * F - (i + 1), (integer > 0) ? 1 : 0);
        }

        set(integer_part);
        add(decimal_part);

        if (v < 0) {
            negate();
        }
    }

    inline __host__ __device__ void set(std::string s) {
        bool minus = s.find('-') == 0;
        if (minus) {
            s = s.substr(1);
        }

        auto dot_index = s.find('.');
        std::string integer = s.substr(0, dot_index);
        std::string decimal = "";
        if (dot_index != std::string::npos) {
            decimal = s.substr(dot_index + 1);
        }

        fixed_point<I, F> tens(1);

        fixed_point<I, F> integer_part;
        for (uint64_t i = 0; i < integer.length(); ++i) {
            fixed_point<I, F> t;
            t.set(integer.at((integer.length() - 1) - i) - '0');
            t.multiply(tens);
            tens.multiply(10ULL);
            integer_part.add(t);
        }

        fixed_point<I, F> decimal_part;
        for (uint64_t i = 0; i < decimal.length(); ++i) {
            uint64_t v = (decimal.at(i) - '0');
            fixed_point<I, F> t;
            for (uint64_t j = 0; j < 32 * F; ++j) {
                v *= 2;
                uint64_t bit_value = v / static_cast<uint64_t>(pow(10, i + 1));
                uint64_t rem_value = v % static_cast<uint64_t>(pow(10, i + 1));
                t.bit_set(32 * F - (j + 1), bit_value);
                v = rem_value;
            }
            decimal_part.add(t);
        }

        set(integer_part);
        add(decimal_part);

        if (minus) {
            negate();
        }
    }

    inline __host__ __device__ void zero_fractional() {
        memset(&data, 0, F * sizeof(uint32_t));
    }

    inline __host__ __device__ void zero_integer() {
        memset(&data[F], 0, I * sizeof(uint32_t));
    }

    inline __host__ __device__ void zero() {
        memset(&data, 0, (I + F) * sizeof(uint32_t));
    }

    //
    // Operations
    //

    inline __host__ __device__ void add(const fixed_point<I, F> &v) {
        uint64_t carry = 0;
        for (uint32_t i = 0; i < I + F; ++i) {
            uint64_t sum = static_cast<uint64_t>(data[i]) + static_cast<uint64_t>(v.data[i]) + carry;
            data[i] = (sum & 0xffffffff);
            carry = (sum >> 32);
        }
    }

    inline __host__ __device__ void add(const uint64_t v) {
        fixed_point<I, F> t(v);
        add(t);
    }

    inline __host__ __device__ void add(const double &v) {
        fixed_point<I, F> t(v);
        add(t);
    }

    inline __host__ __device__ void complement() {
        for (uint32_t i = 0; i < I + F; ++i) {
            data[i] = ~data[i];
        }
    }

    inline __host__ __device__ void multiply(const fixed_point<I, F> &b) {
        fixed_point<2 * I + F, F> accum;
        fixed_point<2 * I + F, F> result;

        fixed_point<2 * I + F, F> a_ext(*this);
        fixed_point<2 * I + F, F> b_ext(b);

        for (uint32_t i = 0; i < 2 * (I + F); ++i) {
            for (uint32_t j = 0; j <= i; ++j) {
                accum.add(static_cast<uint64_t>(a_ext.data[j]) * static_cast<uint64_t>(b_ext.data[i - j]));
            }
            result.data[i] = accum.data[F];

            // Combine: accum.shiftr_32(); accum.zero_fractional();
            for (int32_t i = F; i < 2 * (I + F) - 1; ++i) {
                accum.data[i] = accum.data[i + 1];
            }
            accum.data[2 * (I + F) - 1] = 0;
            memset(&accum.data, 0, sizeof(uint32_t) * F);
        }

        // Combine: result.shiftr(32 * F); set(result);
        memcpy(&data, &result.data[F], sizeof(data));
    }

    inline __host__ __device__ void multiply(const uint64_t &v) {
        fixed_point<I, F> t(v);
        multiply(t);
    }

    inline __host__ __device__ void multiply(const double &v) {
        fixed_point<I, F> t(v);
        multiply(t);
    }

    inline __host__ __device__ void negate() {
        for (uint32_t i = 0; i < I + F; ++i) {
            data[i] = ~data[i];
        }
        fixed_point<I, F> lsb;
        lsb.data[0] = 1;
        add(lsb);
    }

    inline __host__ __device__ void shiftl(const uint64_t v) {
        if ((v % 32) == 0) {
            uint64_t offset = v / 32;
            memmove(&data[offset], &data, sizeof(uint32_t) * (I + F - offset));
            memset(&data, 0, sizeof(uint32_t) * offset);
        }
        else {
            for (int32_t i = I + F - 1; i >= 0; --i) {
                for (int32_t j = 31; j >= 0; --j) {
                    int32_t sbit = i * 32 + j - static_cast<int32_t>(v);
                    uint32_t sbit_value = (sbit >= 0) ? ((data[sbit / 32] & (1 << (sbit % 32))) >> (sbit % 32)) : 0;
                    data[i] = (data[i] & ~(1 << j)) | (sbit_value << j);
                }
            }
        }
    }

    inline __host__ __device__ void shiftr(const uint64_t v) {
        if ((v % 32) == 0) {
            uint64_t offset = v / 32;
            for (int32_t i = 0; i <= I + F - offset; ++i) {
                data[i] = data[i + offset];
            }
            memset(&data[I + F - offset], 0, sizeof(uint32_t) * offset);
        }
        else {
            for (int32_t i = 0; i <= I + F - 1; ++i) {
                for (int32_t j = 0; j <= 31; ++j) {
                    int32_t sbit = i * 32 + j + static_cast<int32_t>(v);
                    uint32_t sbit_value = (sbit < ((I + F) * 32)) ? ((data[sbit / 32] & (1 << (sbit % 32))) >> (sbit % 32)) : 0;
                    data[i] = (data[i] & ~(1 << j)) | (sbit_value << j);
                }
            }
        }
    }
    inline __host__ __device__ void shiftr_32() {
        for (int32_t i = 0; i < I + F - 1; ++i) {
            data[i] = data[i + 1];
        }
        data[I + F - 1] = 0;
    }

public:
    uint32_t data[I + F];
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////