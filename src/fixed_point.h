//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Fractal
// (C)2018-19 Liam Kirton <liam@int3.ws>
//
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<uint32_t I, uint32_t F>
class fixed_point {
public:
    inline __host__ __device__ fixed_point() : data{ 0 } {
    }

    inline __host__ __device__ fixed_point(const fixed_point<I, F> &v) {
        memcpy(&data, &v.data, sizeof(data));
    }

    inline __host__ __device__ fixed_point(const fixed_point<I, F> *v) {
        memcpy(&data, &v->data, sizeof(data));
    }

    template<typename T>
    inline __host__ __device__ fixed_point(const T &v) {
        set(v);
    }

    inline __host__ __device__ fixed_point(const char *v) {
        set(std::string(v));
    }

    //
    // Accessors & Initialisers
    //

    inline __host__ __device__ operator double() {
        return get_double();
    }

    inline __host__ __device__ operator int64_t() {
        return get_integer();
    }

    inline __host__ operator std::string() const {
        return get_string();
    }

    inline __host__ __device__ fixed_point<I, F>& operator=(const fixed_point<I, F>& v) {
        set(v);
        return *this;
    }

    inline __host__ __device__ fixed_point<I, F>& operator+=(const double& v) {
        add(v);
        return *this;
    }

    inline __host__ __device__ fixed_point<I, F>& operator+=(const fixed_point<I, F>& v) {
        add(v);
        return *this;
    }

    inline __host__ __device__ fixed_point<I, F>& operator*=(const double& v) {
        multiply(v);
        return *this;
    }

    inline __host__ __device__ fixed_point<I, F>& operator*=(const fixed_point<I, F>& v) {
        multiply(v);
        return *this;
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

    inline __host__ __device__ uint32_t get_fractional_significant_bit() const {
        for (int32_t i = F - 1; i >= 0; --i) {
            if (data[i] != 0) {
                for (int32_t j = 31; j >= 0; --j) {
                    if (data[i] & (1 << j)) {
                        return 32 * (F - 1 - i) + 32 - j;
                    }
                }
            }
        }
        return 0;
    }

    inline __host__ __device__ double get_double() const {
        fixed_point<I, F> t(*this);
        bool negative = t.negative();
        if (negative) {
            t.negate();
        }

        uint64_t exponent = 1023;
        uint64_t fraction = 0;
        int64_t integer = t.get_integer();

        if (F == 1) {
            fraction = static_cast<const uint64_t>(*reinterpret_cast<const uint32_t *>(&t.data[F - 1]));
        }
        else if (F == 2) {
            fraction = *reinterpret_cast<const uint64_t *>(&t.data[F - 2]);
            if ((integer == 0) && ((fraction & 0xfff0000000000000) == 0)) {
                fraction <<= 12;
                exponent -= 12;
            }
        }
        else {
            fraction = (static_cast<uint64_t>(t.data[F - 2]) << 32) | t.data[F - 3];
            if ((integer == 0) && ((fraction & 0xfff0000000000000) == 0)) {
                fraction = 0;
                for (int32_t i = F - 1; i >= 0; --i) {
                    uint64_t v = t.data[i];
                    if (v != 0) {
                        for (int32_t j = 31; j >= 0; --j) {
                            if (v & (1ULL << j)) {
                                v <<= (32 + 31 - j);
                                fraction |= v;
                                v = ((i - 1) >= 0) ? t.data[i - 1] : 0;
                                v <<= (31 - j);
                                fraction |= v;
                                v = ((i - 2) >= 0) ? t.data[i - 2] : 0;
                                v >>= j + 1;
                                fraction |= v;
                                exponent -= (32 * (F - 1 - i) + (31 - j));
                                break;
                            }
                        }
                        break;
                    }
                }
            }
        }

        double delta;
        double value;
        *reinterpret_cast<uint64_t *>(&delta) = ((exponent & 0x7ff) << 52);
        *reinterpret_cast<uint64_t *>(&value) = ((exponent & 0x7ff) << 52) | (fraction >> 12);

        value = static_cast<double>(integer) + value - delta;
        if (negative) {
            value *= -1.0;
        }
        return value;
    }

    inline __host__ __device__ uint64_t get_fractional() const {
        if (F == 1) {
            return static_cast<const uint64_t>(*reinterpret_cast<const uint32_t *>(&data[F - 1]));
        }
        return *reinterpret_cast<const uint64_t *>(&data[F - 2]);
    }

    inline __host__ __device__ int64_t get_integer() const {
        if (I == 1) {
            return static_cast<const int64_t>(*reinterpret_cast<const int32_t *>(&data[F]));
        }
        return *reinterpret_cast<const int64_t *>(&data[F]);
    }

    inline __host__ std::string get_string() const {
        fixed_point<I, F> t(*this);
        bool negative = t.negative();
        if (negative) {
            t.negate();
        }

        fixed_point<I, F> integer_part(t);
        fixed_point<I, F> decimal_part(t);
        integer_part.zero_decimal();
        decimal_part.zero_integer();

        std::string integer;
        if (integer_part.is_zero()) {
            integer = "0";
        }
        while (!integer_part.is_zero()) {
            uint64_t v = (integer_part.get_integer() % 10);
            integer += static_cast<char>('0' + v);

            fixed_point<I, F> sub_v(v);
            sub_v.negate();
            integer_part.add(sub_v);
            integer_part.multiply(0.1);
        }
        std::reverse(integer.begin(), integer.end());

        std::string decimal;
        for (uint32_t i = 0; i < static_cast<uint32_t>(32 * F * log(2.0) / log(10.0)); ++i) {
            decimal_part.multiply(10ULL);
            decimal += static_cast<char>('0' + (decimal_part.get_integer() % 10));
            decimal_part.zero_integer();
        }

        std::string result;
        if (negative) {
            result += "-";
        }
        result += integer + "." + decimal;
        return result;
    }

    inline __host__ __device__ bool is_zero() const {
        for (uint32_t i = F; i < I + F; ++i) {
            if (data[i] != 0) {
                return false;
            }
        }
        return true;
    }

    inline __host__ __device__ bool negative() const {
        return (data[I + F - 1] & 0x80000000) != 0;
    }

    inline __host__ __device__ void set(const fixed_point<I, F> &v) {
        memcpy(&data, &v.data, sizeof(data));
    }

    template<uint32_t I_X, uint32_t F_X>
    inline __host__ __device__ void set(const fixed_point<I_X, F_X> &v) {
        memset(&data, 0, sizeof(data));

        memcpy(&data[F], &v.data[F_X], sizeof(uint32_t) * ((I <= I_X) ? I : I_X));
        for (int32_t i = 0; i < F; ++i) {
            uint32_t t = ((static_cast<int32_t>(F_X) - 1 - i) >= 0) ? v.data[F_X - 1 - i] : 0;
            data[F - 1 - i] = t;
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
        set(static_cast<int64_t>(integer));

        fixed_point<I, F> decimal_part;
        for (uint32_t i = 0; i < 32 * F; ++i) {
            decimal *= 2;
            decimal = modf(decimal, &integer);
            decimal_part.bit_set(32 * F - (i + 1), (integer > 0) ? 1 : 0);
        }
        add(decimal_part);

        if (v < 0) {
            negate();
        }
    }

    inline __host__ void set(std::string s) {
        std::transform(s.begin(), s.end(), s.begin(), ::tolower);

        auto exp_delim = s.find('e');
        int exp_val = 0;
        if (exp_delim != std::string::npos) {
            exp_val = std::atoi(s.substr(exp_delim + 1).c_str());
            s = s.substr(0, exp_delim);
        }

        bool minus = s.find('-') == 0;
        if (minus) {
            s = s.substr(1);
        }

        auto dot_index = s.find('.');
        if (dot_index == std::string::npos) {
            dot_index = s.length();
        }
        s.erase(std::remove(s.begin(), s.end(), '.'), s.end());

        if (exp_val != 0) {
            int64_t dot_offset = static_cast<int64_t>(dot_index) + exp_val;
            while (dot_offset > static_cast<int64_t>(s.length())) {
                s += '0';
            }
            while (dot_offset <= 0) {
                s = '0' + s;
                dot_offset++;
            }
            dot_index = dot_offset;
        }

        std::string integer = s.substr(0, dot_index);
        std::string decimal = "";
        if (dot_index != std::string::npos) {
            decimal = s.substr(dot_index);
        }

        fixed_point<I + F, I + F> tens(1);
        fixed_point<I + F, I + F> tenths(0.1);

        fixed_point<I + F, I + F> integer_part;
        for (uint64_t i = 0; i < integer.length(); ++i) {
            fixed_point<I + F, I + F> t;
            t.set(integer.at((integer.length() - 1) - i) - '0');
            t.multiply(tens);
            tens.multiply(10ULL);
            integer_part.add(t);
        }

        tens.set(10);

        fixed_point<I + F, I + F> decimal_part;
        for (uint64_t i = 0; i < decimal.length(); ++i) {
            fixed_point<I + F, I + F> v(decimal.at(i) - '0');
            fixed_point<I + F, I + F> t;
            for (uint64_t j = 0; j < 32 * (I + F); ++j) {
                fixed_point<I + F, I + F> v_div;
                v.multiply(2ULL);
                v.multiply(tenths, v_div);

                uint64_t bit_value = v_div.get_integer();
                if (bit_value != 0) {
                    t.bit_set(32 * (I + F) - (j + 1), bit_value);
                }

                v_div.zero_decimal();
                v_div.multiply(tens);
                v_div.negate();
                v.add(v_div);
            }
            tens.multiply(10ULL);
            tenths.multiply(0.1);
            decimal_part.add(t);
        }

        std::memcpy(&data[0], &decimal_part.data[I], sizeof(uint32_t) * F);
        std::memcpy(&data[F], &integer_part.data[I + F], sizeof(uint32_t) * I);

        if (minus) {
            negate();
        }
    }

    inline __host__ __device__ void zero_decimal() {
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
        uint64_t carry = 0;
        for (uint32_t i = 0; i < ((I < 3) ? I : 3); ++i) {
            uint64_t sum = static_cast<uint64_t>(data[F + i]) + ((i < 2) ? static_cast<uint64_t>(reinterpret_cast<const uint32_t *>(&v)[i]) : 0) + carry;
            data[F + i] = (sum & 0xffffffff);
            carry = (sum >> 32);
        }
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

    template<typename T>
    inline __host__ __device__ void multiply(const T &b) {
        uint32_t result[I + F];
        multiply(b, result);
        memcpy(&data, &result, sizeof(data));
    }

    template<typename T>
    inline __host__ __device__ void multiply(const T &b, fixed_point<I, F> &result) {
        multiply(b, result.data);
    }

    inline __host__ __device__ void multiply(const double &b, uint32_t(&result)[I + F]) {
        fixed_point<I, F> t(b);
        multiply(t, result);
    }

    inline __host__ __device__ void multiply(const uint64_t &b, uint32_t(&result)[I + F]) {
        fixed_point<I + F, 0> accum;

        volatile const fixed_point<I, F> *a_p = this; // CUDA Compiler Bug (22/07/2018)
        uint32_t a_ext = (data[(I + F) - 1] & 0x80000000) ? 0xffffffff : 0;

        for (uint32_t i = F; i < I + 2 * F; ++i) {
            for (uint32_t j = 0; j <= i; ++j) {
                const uint32_t l_ix = j;
                const uint32_t r_ix = i - j;
                if ((r_ix == F) || (r_ix == F + 1)) {
                    accum.add(
                        static_cast<uint64_t>((l_ix < (I + F)) ? a_p->data[l_ix] : a_ext) *
                        static_cast<uint64_t>((r_ix == F) ? (b & 0xffffffff) : (b >> 32))
                    );
                }
            }

            // Combine: result.shiftr(32 * F); set(result); accum.shiftr_32();
            if (i >= F) {
                result[i - F] = accum.data[0];
            }
            for (uint32_t j = 0; j < I + F - 1; ++j) {
                accum.data[j] = accum.data[j + 1];
            }
            accum.data[I + F - 1] = 0;
        }
    }

    inline __host__ __device__ void multiply(const fixed_point<I, F> &b, uint32_t(&result)[I + F]) {
        fixed_point<I + F, 0> accum;

        // CUDA Compiler Bug (22/07/2018)
        volatile const fixed_point<I, F> *a_p = this;
        volatile const fixed_point<I, F> *b_p = &b;

        uint32_t a_ext = (data[(I + F) - 1] & 0x80000000) ? 0xffffffff : 0;
        uint32_t b_ext = (b.data[(I + F) - 1] & 0x80000000) ? 0xffffffff : 0;

        for (uint32_t i = 0; i < I + 2 * F; ++i) {
            for (uint32_t j = 0; j <= i; ++j) {
                const uint32_t l_ix = j;
                const uint32_t r_ix = i - j;
                accum.add(
                    static_cast<uint64_t>((l_ix < (I + F)) ? a_p->data[l_ix] : a_ext) *
                    static_cast<uint64_t>((r_ix < (I + F)) ? b_p->data[r_ix] : b_ext)
                );
            }

            // Combine: result.shiftr(32 * F); set(result); accum.shiftr_32();
            if (i >= F) {
                result[i - F] = accum.data[0];
            }
            for (uint32_t j = 0; j < I + F - 1; ++j) {
                accum.data[j] = accum.data[j + 1];
            }
            accum.data[I + F - 1] = 0;
        }
    }

    inline __host__ __device__ void negate() {
        uint64_t carry = 1;
        for (uint32_t i = 0; i < I + F; ++i) {
            const uint32_t neg = ~data[i];
            uint64_t sum = static_cast<uint64_t>(neg) + carry;
            data[i] = (sum & 0xffffffff);
            carry = (sum >> 32);
        }
    }

    inline __host__ __device__ void shiftl(const uint64_t v) {
        if ((v % 32) == 0) {
            uint64_t offset = v / 32;
            for (uint32_t i = 0; i < I + F; ++i) {
                data[i] = (i < offset) ? 0 : data[i - offset];
            }
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

    inline __host__ __device__ void shiftl_32() {
        memmove(&data[1], &data, sizeof(uint32_t) * (I + F - 1));
        data[0] = 0;
    }

    inline __host__ __device__ void shiftr(const uint64_t v) {
        if ((v % 32) == 0) {
            uint64_t offset = v / 32;
            for (uint32_t i = 0; i <= I + F - offset; ++i) {
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

template<uint32_t I, uint32_t F>
std::ostream& operator<<(std::ostream &o, const fixed_point<I, F> &v) {
    o << static_cast<std::string>(v);
    return o;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
