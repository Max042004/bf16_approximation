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

static bf16_t bf16_sin(bf16_t a, int *record_k)
{
	uint16_t sign = (a.bits >> 15) & 0x1;
	a.bits &= 0x7FFF;
	int32_t k = 0;
	bf16_t result;

	// a so small, let sin(a) = a
	if ((a.bits & 0x7FFF) < 0b0011111000010000) {
		if (sign) a.bits ^= 0x8000;
		return a;
	}

	// range reduction	
	if ((a.bits & 0x7FFF) >= 0b0011111111001010) {
		a = fp32_to_bf16(trig_red_slowpath_f(bf16_to_fp32(a), &k));
		printf("Angle after range reduction: %f\n", bf16_to_fp32(a));
	}
	
	// sin(x)
	bf16_t sin_x = chebyshev_sin_6terms(bf16_to_fp32(a));

	// cos(x) = sqrt(1-sin^2(x))
	float cos_a = pi_over_two_float - bf16_to_fp32(a);
	bf16_t cos_x = chebyshev_sin_6terms(cos_a);

	// k mod 4
	unsigned mod_k = k % 4;
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
	printf("sin(r) = %f, cos(r) = %f, k = %d, mod_k = %d\n", bf16_to_fp32(sin_x), bf16_to_fp32(cos_x),k, mod_k);
	if (sign) result.bits ^= 0x8000;
	return result;
}
