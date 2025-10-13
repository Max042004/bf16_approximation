/*
	bfloat quotient = bf16_to_fp32(a) / bf16_to_fp32(pi_over_two_bits);
	*k = floorf(quotient);
 * Bfloat16: 0bxxxxxxxxxxxxxxxx
 *	       ||______||_____|
 *             |    |      |
 *             1    |      7
 *             v    8      v
 *            sign  v     mantissa
 *               exponent
 *
 * (-1)^sign * 2(exponent-127) * 1.mantissa
 */

#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <stdlib.h>

typedef struct { uint16_t bits;} bf16_t;

#define BF16_EXP_BIAS 127
#define BF16_SIGN_MASK 0x8000U
#define BF16_EXP_MASK 0x7F80U
#define BF16_MANT_MASK 0x007FU

#define BF16_NAN() ((bf16_t) {.bits = 0x7FC0})
#define BF16_ZERO() ((bf16_t) {.bits = 0x0000})
	
bf16_t bf16_one = (bf16_t) { .bits = 0x3F80}; // BF16_ONE
bf16_t bf16_two = (bf16_t) { .bits = 0x4000}; // BF16_TWO
bf16_t bf16_four = (bf16_t) { .bits = 0x4080}; // BF16_FOUR
bf16_t sign = (bf16_t) {.bits = 0x8000};
bf16_t pi_over_two_bits = (bf16_t) {.bits = 0b0011111111001001};
float pi_over_two_float = 1.5707962512969970703125;

static inline bool bf16_isnan(bf16_t a)
{
    return ((a.bits & BF16_EXP_MASK) == BF16_EXP_MASK) &&
           (a.bits & BF16_MANT_MASK);
}

static inline bool bf16_isinf(bf16_t a)
{
    return ((a.bits & BF16_EXP_MASK) == BF16_EXP_MASK) &&
           !(a.bits & BF16_MANT_MASK);
}

static inline bool bf16_iszero(bf16_t a)
{
    return !(a.bits & 0x7FFF);
}

// remember to deal with special cases
static inline bf16_t fp32_to_bf16(float val)
{
	uint32_t f32bits;
	memcpy(&f32bits, &val, sizeof(float));
	
	// Infinity or NaN
	if ((f32bits >> 23 & 0xFF) == 0xFF) 
		return (bf16_t) {.bits = f32bits >> 16 & 0xFFFF}; // &0xFFFF can be omitted
	
	// Normalized
	f32bits += (f32bits >> 16 & 1) + 0x7FFF;
	return (bf16_t) {.bits = f32bits >> 16};	
}

static inline float bf16_to_fp32 (bf16_t val)
{
	uint32_t f32bits = ((uint32_t) val.bits) << 16;
	float result;
	memcpy(&result, &f32bits, sizeof(float));
	return result;
}

static inline unsigned clz(uint32_t x)
{
    int n = 32, c = 16;
    do {
        uint32_t y = x >> c;
        if (y) {
            n -= c;
            x = y;
        }
        c >>= 1;
    } while (c);
    return n - x; // after interation, x still have one last bit not count, so n = n-x
}

/*
 * bf16_add
 * 1. Before adding, first deal with a or b is special case(NaN, Infinity, 0).
 * 2. After 1., remaining a and b are all normalized or denormalized. Calculate a + b.
 *	Ex. 2^15 * 1.1101 + (-1) * 2^12 * 1.11 = 2^15 * (1.1101 - 0.00111) = 2^15 * 1.0011
 *	Calculate exponent first, then mantissa.
 * 3. deal with overflow(return Inifinty) and underflow(return 0)
 *
 */


static inline bf16_t bf16_add (bf16_t a, bf16_t b)
{
	uint16_t sign_a = a.bits >> 15 & 0x1, sign_b = b.bits >> 15 & 1;
	int16_t exp_a = a.bits >> 7 & 0xFF, exp_b = b.bits >> 7 & 0xFF;
	uint16_t mant_a = a.bits & 0x7F, mant_b = b.bits & 0x7F;

	// Infinity and NaN
	if (exp_a == 0xFF) {
		if (mant_a) return a;
		if (exp_b == 0xFF) 
			return (mant_b || sign_a == sign_b)?b:BF16_NAN();
		return a;
	}

	// if a is normal/denormal, but b is infinity/NaN
	if (exp_b == 0xFF) return b;

	// if a == 0, b == 0
	if (!exp_a && !mant_a) return b;
	if (!exp_b && !mant_b) return a;

	// if a, b is normal
	if (exp_a) mant_a |= 0x80;
	if (exp_b) mant_b |= 0x80;

	int16_t exp_diff = exp_a - exp_b;
	uint16_t result_sign;
	int16_t result_exp;
	uint32_t result_mant;

	// deal with result of exp
	if(exp_diff > 0) {
		result_exp = exp_b;
		if (exp_diff > 8) return a;
		mant_a <<= exp_diff;
	} else if (exp_diff < 0) {
		result_exp = exp_a;
		if (exp_diff < -8) return b;
		mant_b <<= -exp_diff;
	}
	else result_exp = exp_a;

	if (sign_a == sign_b) {
		result_sign = sign_a;
		result_mant = (uint32_t) mant_a + mant_b;
		uint32_t lz = clz(result_mant);
		for (unsigned i = 0; i < 32-lz-8; i++){
			result_mant >>= 1;
			if (++result_exp >= 255) return BF16_NAN();
		}
	}
	else {
		if (mant_a >= mant_b) {
			result_sign = sign_a;
			result_mant = mant_a - mant_b;
		}
		else {
			result_sign = sign_b;
			result_mant = mant_b - mant_a;
		}
		if (!result_mant) return BF16_ZERO();
		if (result_mant < 0x80) {
			while (!(result_mant & 0x80)) {
				result_mant <<= 1;
				// here, we flush out denormalized value to zero, only return normalized. 
				if (--result_exp <= 0) return BF16_ZERO();
			}
		}
		else {
			uint32_t lz = clz(result_mant);
			for (unsigned i = 0; i < 32-lz-8; i++){
				result_mant >>= 1;
				if (++result_exp >= 255) return BF16_NAN();
			}
		}
	}
	return (bf16_t) {
		.bits = result_sign << 15 | (result_exp & 0xFF) << 7 | result_mant & 0x7F, 
	};
}

static inline bf16_t bf16_sub(bf16_t a, bf16_t b)
{
	b.bits ^= 0x8000U;
	return bf16_add(a, b);
}

/*
 * bf16_sqrt
 * 1. deal with special case
 * 	sqrt(+-0) = 0
 *	sqrt(Infinity) = Infinity
 *	sqrt(NaN) = NaN
 *	sqrt(-Infinity) = NaN
 * 	sqrt(-x) = NaN
 *	sqrt(denormalized) = 0 (this is self defined)
 * 2. sqrt(normalized)
 *	first, for exp is even, result exp = exp/2 
 *	for exp is odd, result exp = (exp+1)/2, and mant must right shift 1.
 *	Second, using binary search to find mantissa sqrt.
 */
static inline bf16_t bf16_sqrt(bf16_t a)
{
	uint16_t sign = a.bits >> 15 & 1;
	uint16_t exp = a.bits >> 7 & 0xFF;
	uint16_t mant = a.bits & 0x7F;

	// sqrt(0) = 0
	if (!exp && !mant) return BF16_ZERO();
	
	// sqrt(+-Infinity, NaN)
	if (exp == 0xFF) {
		if (mant) return a;
		if (sign) return BF16_NAN();
		return a;
	}

	// -x
	if (sign) return BF16_NAN();

	// flash denormal to zero
	if (!exp) return BF16_ZERO();

	uint16_t e = exp - BF16_EXP_BIAS; // 127
	uint16_t result_sign, result_exp, result_mant;

	// to here, except a is normalized, other special cases are all out.
	if (exp & 0x1) {
		exp -= 1;
		mant <<= 1;
		result_exp = exp >> 1;
	} else {
		result_exp = exp >> 1;
	}
	
	// binary search finding sqrt(mant)
	uint16_t low, high;
}

bf16_t bf16_mul2(bf16_t a)
{
	uint16_t sign = a.bits >> 15 & 0x1;
	uint16_t exp = a.bits >> 7 & 0xFF;
	uint16_t mant = a.bits & 0x7F;

	if (exp >= 0xFE) return BF16_NAN();
	
	exp += 1;

	return (bf16_t) {
        	.bits = (sign << 15) | ((exp & 0xFF) << 7) |
                	(mant & 0x7F),
    	};
}

static inline bf16_t bf16_mul(bf16_t a, bf16_t b)
{
    uint16_t sign_a = (a.bits >> 15) & 1;
    uint16_t sign_b = (b.bits >> 15) & 1;
    int16_t exp_a = ((a.bits >> 7) & 0xFF);
    int16_t exp_b = ((b.bits >> 7) & 0xFF);
    uint16_t mant_a = a.bits & 0x7F;
    uint16_t mant_b = b.bits & 0x7F;

    uint16_t result_sign = sign_a ^ sign_b;

    if (exp_a == 0xFF) {
        if (mant_a)
            return a;
        if (!exp_b && !mant_b)
            return BF16_NAN();
        return (bf16_t) {.bits = (result_sign << 15) | 0x7F80};
    }
    if (exp_b == 0xFF) {
        if (mant_b)
            return b;
        if (!exp_a && !mant_a)
            return BF16_NAN();
        return (bf16_t) {.bits = (result_sign << 15) | 0x7F80};
    }
    if ((!exp_a && !mant_a) || (!exp_b && !mant_b))
        return (bf16_t) {.bits = result_sign << 15};

    int16_t exp_adjust = 0;
    if (!exp_a) {
        while (!(mant_a & 0x80)) {
            mant_a <<= 1;
            exp_adjust--;
        }
        exp_a = 1;
    } else
        mant_a |= 0x80;
    if (!exp_b) {
        while (!(mant_b & 0x80)) {
            mant_b <<= 1;
            exp_adjust--;
        }
        exp_b = 1;
    } else
        mant_b |= 0x80;

    uint32_t result_mant = (uint32_t) mant_a * mant_b;

    int32_t result_exp = (int32_t) exp_a + exp_b - BF16_EXP_BIAS + exp_adjust;

    if (result_mant & 0x8000) {
        result_mant = (result_mant >> 8) & 0x7F;
        result_exp++;
    } else
        result_mant = (result_mant >> 7) & 0x7F;

    if (result_exp >= 0xFF)
        return (bf16_t) {.bits = (result_sign << 15) | 0x7F80};
    if (result_exp <= 0) {
        if (result_exp < -6)
            return (bf16_t) {.bits = result_sign << 15};
        result_mant >>= (1 - result_exp);
        result_exp = 0;
    }

    return (bf16_t) {.bits = (result_sign << 15) | ((result_exp & 0xFF) << 7) |
                             (result_mant & 0x7F)};
}

static inline bf16_t bf16_div(bf16_t a, bf16_t b)
{
    uint16_t sign_a = (a.bits >> 15) & 1;
    uint16_t sign_b = (b.bits >> 15) & 1;
    int16_t exp_a = ((a.bits >> 7) & 0xFF);
    int16_t exp_b = ((b.bits >> 7) & 0xFF);
    uint16_t mant_a = a.bits & 0x7F;
    uint16_t mant_b = b.bits & 0x7F;

    uint16_t result_sign = sign_a ^ sign_b;

    if (exp_b == 0xFF) {
        if (mant_b)
            return b;
        /* Inf/Inf = NaN */
        if (exp_a == 0xFF && !mant_a)
            return BF16_NAN();
        return (bf16_t) {.bits = result_sign << 15};
    }
    if (!exp_b && !mant_b) {
        if (!exp_a && !mant_a)
            return BF16_NAN();
        return (bf16_t) {.bits = (result_sign << 15) | 0x7F80};
    }
    if (exp_a == 0xFF) {
        if (mant_a)
            return a;
        return (bf16_t) {.bits = (result_sign << 15) | 0x7F80};
    }
    if (!exp_a && !mant_a)
        return (bf16_t) {.bits = result_sign << 15};

    if (exp_a)
        mant_a |= 0x80;
    if (exp_b)
        mant_b |= 0x80;

    uint32_t dividend = (uint32_t) mant_a << 15;
    uint32_t divisor = mant_b;
    uint32_t quotient = 0;

    for (int i = 0; i < 16; i++) {
        quotient <<= 1;
        if (dividend >= (divisor << (15 - i))) {
            dividend -= (divisor << (15 - i));
            quotient |= 1;
        }
    }

    int32_t result_exp = (int32_t) exp_a - exp_b + BF16_EXP_BIAS;

    if (!exp_a)
        result_exp--;
    if (!exp_b)
        result_exp++;

    if (quotient & 0x8000)
        quotient >>= 8;
    else {
        while (!(quotient & 0x8000) && result_exp > 1) {
            quotient <<= 1;
            result_exp--;
        }
        quotient >>= 8;
    }
    quotient &= 0x7F;

    if (result_exp >= 0xFF)
        return (bf16_t) {.bits = (result_sign << 15) | 0x7F80};
    if (result_exp <= 0)
        return (bf16_t) {.bits = result_sign << 15};
    return (bf16_t) {.bits = (result_sign << 15) | ((result_exp & 0xFF) << 7) |
                             (quotient & 0x7F)};
}

static bf16_t bf16_floor(bf16_t a)
{
	uint16_t sign = (a.bits >> 15) & 0x1;
	uint16_t exp = (a.bits >> 7) & 0xFF;
	uint16_t mant = a.bits & 0x7F;
	int16_t unbiased_exp = exp - 127;

	if (unbiased_exp < 7) {
		mant >>= 7 - unbiased_exp;
		mant <<= 7 - unbiased_exp;
	}
	
	return (bf16_t) {.bits = (sign << 15) | ((exp & 0xFF) << 7) | (mant & 0x7F)};
}

static uint32_t bf16_to_uint32(bf16_t a)
{
	uint16_t exp = (a.bits >> 7) & 0xFF;
	uint32_t mant = a.bits & 0x7F;
	mant |= 0x80;
	int16_t unbiased_exp = exp - 127;

	if (unbiased_exp >= 7) mant <<= (unbiased_exp - 7);
	if (unbiased_exp >= 0 && unbiased_exp < 7) mant >>= (7 - unbiased_exp);

	return mant;
}

static inline void tylor_sin(bf16_t *result, bf16_t *it, bf16_t *it_deno, bf16_t *it_mole, bf16_t *it_n)
{	
	*it_deno = bf16_add(bf16_mul(bf16_four, bf16_mul(*it_n, *it_n)), bf16_mul(bf16_two, *it_n));
	*it = bf16_mul(*it, *it_mole);
	*it = bf16_div(*it, *it_deno);
	it->bits ^= sign.bits;
	*it_n = bf16_add(*it_n, bf16_one);
	
	*result = bf16_add(*result, *it);
}

static bf16_t tylor_sin_loop_unrool(bf16_t a)
{
	bf16_t result = a, it_mole = bf16_mul(a, a), it_deno, it = a, it_n = bf16_one;

	tylor_sin(&result, &it, &it_deno, &it_mole, &it_n);
	tylor_sin(&result, &it, &it_deno, &it_mole, &it_n);
	tylor_sin(&result, &it, &it_deno, &it_mole, &it_n);
	tylor_sin(&result, &it, &it_deno, &it_mole, &it_n);
	tylor_sin(&result, &it, &it_deno, &it_mole, &it_n);
	tylor_sin(&result, &it, &it_deno, &it_mole, &it_n);
	tylor_sin(&result, &it, &it_deno, &it_mole, &it_n);

	return result;
}

static bf16_t chebyshev_sin_5terms(bf16_t a)
{
	float para = bf16_to_fp32(a);
	float x_2 = para*para;
	float x_3 = x_2*para;
	float x_4 = x_3*para;
	float result = 0.0368*x_4-0.2311*x_3+0.0489*x_2+0.9867*para+0.0005833;
	// degress 5 to small, so ignore
	return fp32_to_bf16(result);
}

static bf16_t chebyshev_sin_6terms(float a)
{
	float para = a;
	float result;
	result = 0.0102485642486;
	result -= (0.000960664234092 * para);
	result *= para;
	result -= 0.0019022700092;
	result *= para;
	result -= 0.165722549388;
	result *= para;
	result -= 0.000206058500652;
	result *= para;
	result += 1.00001322248;
	result *= para;
	return fp32_to_bf16(result);
}

/* 190 bits of 2/pi for Payne-Hanek style argument reduction. */
static const unsigned int two_over_pi_f [] = 
{
    0x00000000,
    0x28be60db,
    0x9391054a,
    0x7f09d5f4,
    0x7d4d3770,
    0x36d8a566,
    0x4f10e410
};

float trig_red_slowpath_f (float a, int *quadrant)
{
    unsigned long long int p;
    unsigned int ia, hi, mid, lo, i;
    int e, q;
    float r;

    ia = (unsigned int)(fabsf (frexpf (a, &e)) * 0x1.0p32f);

    /* extract 96 relevant bits of 2/pi based on magnitude of argument */ 
    i = (unsigned int)e >> 5;
    e = (unsigned int)e & 31;

    if (e) {
        hi  = (two_over_pi_f [i+0] << e) | (two_over_pi_f [i+1] >> (32 - e));
        mid = (two_over_pi_f [i+1] << e) | (two_over_pi_f [i+2] >> (32 - e));
        lo  = (two_over_pi_f [i+2] << e) | (two_over_pi_f [i+3] >> (32 - e));
    } else {
        hi  = two_over_pi_f [i+0];
        mid = two_over_pi_f [i+1];
        lo  = two_over_pi_f [i+2];
    }

    /* compute product x * 2/pi in 2.62 fixed-point format */
    p = (unsigned long long int)ia * lo;
    p = (unsigned long long int)ia * mid + (p >> 32);
    p = ((unsigned long long int)(ia * hi) << 32) + p;

    /* round quotient to nearest */
    q = (int)(p >> 62);                // integral portion = quadrant<1:0>
    p = p & 0x3fffffffffffffffULL;     // fraction
    if (p & 0x2000000000000000ULL) {   // fraction >= 0.5
        p = p - 0x4000000000000000ULL; // fraction - 1.0
        q = q + 1;
    }

    /* compute remainder of x / (pi/2) */
    double d;

    d = (double)(long long int)p;
    d = d * 0x1.921fb54442d18p-62; // 1.5707963267948966 * 0x1.0p-62
    r = (float)d;
    if (a < 0.0f) {
        r = -r;
        q = -q;
    }

    printf("After payne_hanek\n");
    *quadrant = q;
    return r;
}

// 200 bits for 2/pi
static const uint8_t two_over_pi_byte[27] = {
    0xA2, 0xF8, 0xC1, 0xB7, 0x27, 0x22, 0x0A, 0x94,
    0xFE, 0x13, 0xA7, 0x55, 0xF4, 0x3E, 0xA6, 0xD7,
    0xC0, 0x6D, 0xB1, 0x4A, 0xCC, 0x9E, 0x21, 0xC8,
    0x20, 0xFF, 0x28
};

typedef union {
	double f;
	uint64_t bits;
} fp64_bits;

static bf16_t quot(bf16_t a)
{
	// a*(2/pi), using fixed point Q2.30
	uint16_t sign = (a.bits >> 15) & 0x1;
	uint16_t exp = (a.bits >> 7) & 0xFF;
	uint32_t mant = a.bits & 0x7F;
	int16_t unbiased_exp = exp - 127;

	// Q2.30
	mant |= 0x80;
	mant <<= 23;
	uint32_t two_over_pi_bits = (two_over_pi_byte[0]<<22) | (two_over_pi_byte[1]<<14)
	    | (two_over_pi_byte[2]<<6) | (two_over_pi_byte[3]>>2);

	uint32_t quo = ((uint64_t)mant * two_over_pi_bits) >> 30;
	if (quo >> 30 == 0) {
		exp -= 1;
		quo <<= 1;
	}

    	return (bf16_t) {.bits = (sign << 15) | ((exp & 0xFF) << 7) |
                             ((quo >> 23) & 0x7F)};
}

static bf16_t mod_range_reduc(bf16_t *k, bf16_t a)
{
	bf16_t quotient = bf16_div(a, pi_over_two_bits);
	*k = bf16_floor(quotient);
	a = bf16_sub(a, bf16_mul(*k, pi_over_two_bits));

	printf("\nk: %f, after naive mod reduction: %f\n",bf16_to_fp32(*k) ,bf16_to_fp32(a));
	return a;
}

/*
 * C = C1+C2+C3
 * x* = x-kC = x-kC1-kC2-kC3 = ((x-kC1)-kC2)-kC3
 */
static bf16_t cody_waite_reduc(bf16_t *k, bf16_t a)
{
	bf16_t c_big = pi_over_two_bits;
	bf16_t c_med = (bf16_t) { .bits = 0b0011100111111100};
	bf16_t c_sm = (bf16_t) {.bits = 0b0011011001010100};
	*k = quot(a);
	*k = bf16_floor(*k);
	a = bf16_sub(bf16_sub(bf16_sub(a, bf16_mul(*k, c_big)), bf16_mul(*k, c_med)), bf16_mul(*k, c_sm));
	
	printf("\nk: %f, after cody-waite reduction: %f\n",bf16_to_fp32(*k) ,bf16_to_fp32(a));
	return a;
}

static bf16_t bf16_sin(bf16_t a, int *record_k)
{
	bf16_t k;
	int32_t kpayne;
	uint16_t isPayne = 0;
	if ((a.bits & 0x7FFF) < 0b0011111000010000) {
		printf("directly return a\n");
		return a;
	} 
	if ((a.bits & 0x7FFF) >= 0b0011111111001010) {
		a = fp32_to_bf16(trig_red_slowpath_f(bf16_to_fp32(a), &kpayne));
		isPayne = 1;
	}
	
	// sin(x)
	printf("angle after range reduction: %f\n", bf16_to_fp32(a));
	bf16_t result = chebyshev_sin_6terms(bf16_to_fp32(a));
	bf16_t sin_x = result;

	// cos(x) = sqrt(1-sin^2(x))
	//bf16_t cos_a = bf16_add(pi_over_two_bits, a);
	float cos_a = pi_over_two_float + bf16_to_fp32(a);
	result = chebyshev_sin_6terms(cos_a);
	bf16_t cos_x = result;

	// k mod 4
	uint32_t k_int;
	if (isPayne){
		k_int = (uint32_t) kpayne;
	}
	else {
		k_int = bf16_to_uint32(k); 
	}
	unsigned mod_k = k_int % 4;
	switch (mod_k) {
		case 0:
			result = sin_x;
			break;
		case 1:
			result = cos_x;
			break;
		case 2:
			result = (bf16_t) {.bits = sin_x.bits ^ 0x8000};
			break;
		case 3:
			result = (bf16_t) {.bits = cos_x.bits ^ 0x8000};
			break;
	}
	
	*record_k = mod_k;
	printf("sin(r) = %f, cos(r) = %f, k_int = %d, mod_k = %d\n", bf16_to_fp32(sin_x), bf16_to_fp32(cos_x),k_int, mod_k);
	return result;
}

int main()
{
	FILE *f = fopen("./log.txt", "w");
	FILE *fk = fopen("./klog.txt", "w");
	
	/* x between [pi/2, 0] 
	 * bf16_t i = (bf16_t){.bits = 0b0011111111001001}; // Initial
	int j = 0;
	for (j; j < 450; j++){
		printf("%f\n", bf16_to_fp32(i));
		float glibc_sin = bf16_to_fp32(fp32_to_bf16(sinf(bf16_to_fp32(i))));
		float payne = bf16_to_fp32(bf16_sin(i));
		printf("i = %f, bf16_sin = %f ,glibc_sin = %f\n", bf16_to_fp32(i), payne, glibc_sin);
		fprintf(f, "%f\n",fabs(payne-glibc_sin));
		i.bits--;
	}*/

	// x larger than pi/2, need range reduction
	bf16_t i = (bf16_t){.bits = 0b0011111111001010}; // Initial
	int j = 0, record_k;
	for (j; j < 16310; j++){
		float glibc_sin = bf16_to_fp32(fp32_to_bf16(sinf(bf16_to_fp32(i))));
		float payne = bf16_to_fp32(bf16_sin(i, &record_k));
		printf("i = %f, bf16_sin = %f ,glibc_sin = %f\n\n", bf16_to_fp32(i), payne, glibc_sin);
		if (fabs(payne-glibc_sin) > 0) fprintf(fk, "%d\n", record_k);
		fprintf(f, "%f\n",fabs(payne-glibc_sin));
		i.bits++;
	}

	fclose(f);
	return 0;
}
