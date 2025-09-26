/*
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
bf16_t two_over_pi_bits = (bf16_t) {.bits = 0b0011111111001001};

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

/*static uint32_t bf16_to_uint32(bf16_t a) {
    uint32_t fbits = (uint32_t)a.bits << 16;
    float f;
    memcpy(&f, &fbits, sizeof f);

    if (isnan(f)) {
        return 0;
    }
    if (isinf(f)) {
        return (f > 0.0f) ? UINT32_MAX : 0;
    }
    if (f <= 0.0f) return 0;

    // 如果 f 超出 uint32 範圍，返回上界 
    if (f >= (float)UINT32_MAX) return UINT32_MAX;

    return (uint32_t)f; 
}*/

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

	return result;
}

// 200 bits for 2/pi
static const uint8_t two_over_pi_byte[27] = {
    0xA2, 0xF8, 0xC1, 0xB7, 0x27, 0x22, 0x0A, 0x94,
    0xFE, 0x13, 0xA7, 0x55, 0xF4, 0x3E, 0xA6, 0xD7,
    0xC0, 0x6D, 0xB1, 0x4A, 0xCC, 0x9E, 0x21, 0xC8,
    0x20, 0xFF, 0x28
};

typedef union {
	float f;
	uint32_t bits;
} fp32_bits;

static bf16_t payne_hanek_reduc(bf16_t *k, bf16_t a)
{
	bf16_t quotient = bf16_div(a, two_over_pi_bits);
	*k = bf16_floor(quotient);

	uint16_t exp = (a.bits >> 7) & 0xFF;
	int16_t unbiased_exp = exp - 127;
	uint16_t p = 8;
	uint16_t n = 8; // 7 bit mantissa + 1 implicit bit


	//bf16_t bf16_one = (bf16_t) { .bits = 0x3F80}; // BF16_ONE
	// X = x * 2^n-e-1
	int16_t tmp = (n-unbiased_exp-1) << 7;
	if (tmp > 0) a.bits += tmp;
	else a.bits -= abs(tmp);
	bf16_t val = a;
	uint32_t tailx = bf16_to_uint32(val);


	/*
	   C = 0.v0 v1 v2 v3 v4...
	         -1 -2 -3 -4 -5...
	   whole bits are divide into three parts Left(e,p), Med(e,p), Right(e,p)
	   which only Med(e,p) influence the range reduction.
	 */
	int16_t start_bit = -1;
	// check how many bits for Left(e,p) 
	int16_t left_last = n-unbiased_exp+1;
	if (left_last < -1) start_bit = left_last - 1;

	// total bits for Med(e,p) = 2n + p + 2
	uint16_t last_bit = ((n << 1) + p + 2) + abs(start_bit) - 1;
	// align for array index
	last_bit -= 1;
	start_bit += 1; 
	uint16_t byte_start = abs(start_bit) / 8;
	uint16_t byte_last = last_bit / 8;
	uint16_t mod_start = abs(start_bit) % 8;
	uint16_t mod_last = last_bit % 8;

	uint64_t med = 0x0;
	for (unsigned int i = byte_start, j = byte_last-byte_start; i <= byte_last; i++, j--) {
		med |= ((uint64_t) two_over_pi_byte[i] << (j<<3)); // here, it will add unneeded bits
	}

	uint16_t med_clz = clz(med);
	// flush out additional unneeded bits
	med <<= (med_clz + mod_start);
	med >>= (med_clz + mod_start);
	med >>= (7 - mod_last);

	// Med*tailx*2^{-16-p}, using uint32_t to store
	uint64_t medmulx = med * tailx;
	uint64_t tmp_result  =  medmulx << (64-(n<<1)-p-1);
	uint32_t frac = (uint32_t) (tmp_result >> (64-(n<<1)-p-1)); // only retain 2n+p+1 bits

	fp32_bits fp32_frac = (fp32_bits) {.bits = 0x0};
	fp32_frac.bits = fp32_frac.bits | 0x3F800000 | ((frac >> ((n<<1)+p+1-23)) & 0x7FFFFF); 
	fp32_frac.f -= 1;// frac = 1.xxxxxx - 1
	float pi_over_twof = 1.5707963;
	float result =  pi_over_twof * fp32_frac.f;

	a = fp32_to_bf16(result);
	printf("\nk: %f, after payne_hanek reduction: %f\n",bf16_to_fp32(*k) ,bf16_to_fp32(a));
	return a;
}

static bf16_t mod_range_reduc(bf16_t *k, bf16_t a)
{
	bf16_t quotient = bf16_div(a, two_over_pi_bits);
	*k = bf16_floor(quotient);
	a = bf16_sub(a, bf16_mul(*k, two_over_pi_bits));

	printf("\nk: %f, after naive mod reduction: %f\n",bf16_to_fp32(*k) ,bf16_to_fp32(a));
	return a;
}

static bf16_t cody_waite_reduc(bf16_t *k, bf16_t a)
{
	bf16_t c_big = two_over_pi_bits;
	bf16_t c_small = (bf16_t) { .bits = 0b0011100111111110};
	bf16_t quotient = bf16_div(a, c_big);
	*k = bf16_floor(quotient);
	a = bf16_sub(bf16_sub(a, bf16_mul(*k, c_big)), bf16_mul(*k, c_small));
	
	printf("\nk: %f, after cody-waite reduction: %f\n",bf16_to_fp32(*k) ,bf16_to_fp32(a));
	return a;
}

/* using tylor expansion
 * sin(x) = x - x^3/3! + x^5/5! - x^7/7! +...
 * iteration value: (-1)^n * x^2 / (4n^2+2*n)
 *
 * optimize:
 * loop unrooling, avoid unneeded calculation for too small tylor value.
 */
static bf16_t bf16_sin(bf16_t a)
{
	// TODO: judge is a out of rage -pi/2 to +pi/2. current judgment not good
	// Using Cody-Waite's Method
	bf16_t k, c_big;
	uint32_t exp = (a.bits >> 7) & 0xFF;
	if (exp >= 0x80) {
		if (exp > 0x84) a = payne_hanek_reduc(&k, a);
		else a = cody_waite_reduc(&k, a);
		//a = mod_range_reduc(&k, a);
	}
	
	// sin(x)
	bf16_t result = tylor_sin_loop_unrool(a);
	bf16_t sin_x = result;

	// cos(x) = sqrt(1-sin^2(x))
	a = bf16_add(two_over_pi_bits, a); // sin(pi/2 + x) = cos(x)
	result = tylor_sin_loop_unrool(a);
	bf16_t cos_x = result;

	// k mod 4
	uint32_t k_int = bf16_to_uint32(k); 
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
	
	printf("sin(r) = %f, cos(r) = %f, k_int = %d, mod_k = %d\n", bf16_to_fp32(sin_x), bf16_to_fp32(cos_x),k_int, mod_k);
	return result;
}

int main()
{
	bf16_t a = {.bits = 0x7fff};
	bf16_t c = {.bits = 0x7aff};
	bf16_t b = bf16_mul2(a);
	bf16_t d = bf16_mul2(c);

	bf16_t x = {.bits = 0x4000};
	bf16_t arr_bf16[13];
	arr_bf16[0] = (bf16_t){.bits = 0x3f80}; // 1
        arr_bf16[1] = (bf16_t){.bits = 0x4000}; // 2
	arr_bf16[2] = (bf16_t){.bits = 0b0100000100100000}; // 10
	arr_bf16[3] = (bf16_t){.bits = 0b0100001011001000}; // 100
	arr_bf16[4] = (bf16_t){.bits = 0}; // 0
	arr_bf16[5] = (bf16_t){.bits = 0b0100000001001001}; // pi
	arr_bf16[6] = (bf16_t){.bits = 0x4100}; // 8
	arr_bf16[7] = (bf16_t){.bits = 0x410a}; // (11/4)pi
	arr_bf16[8] = (bf16_t){.bits = 0b0100001010110100}; // 90
	arr_bf16[9] = (bf16_t){.bits = 0b0100001010000000}; // 64
	arr_bf16[10] = (bf16_t){.bits = 0b0100001010000010}; // 65

	for(int i = 0; i < 13; i++){
		printf("x = %f, sin(x) = %f\n\n", bf16_to_fp32(arr_bf16[i]), bf16_to_fp32(bf16_sin(arr_bf16[i])));
	}
	return 0;
}
