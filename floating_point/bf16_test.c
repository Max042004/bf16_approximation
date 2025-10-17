#include <stdio.h>
#include <stdlib.h>
#include "bf16.h"
#include <math.h>

int main()
{
	FILE *f1 = fopen("./log/range1.txt", "w");
	FILE *f2 = fopen("./log/range2.txt", "w");
	FILE *f3 = fopen("./log/range3.txt", "w");
	
	// x between [pi/2, 0] 
	bf16_t i = (bf16_t){.bits = 0b0011111111001001}; // Initial
	int j = 0, record_k, error_count = 0;
	float difference, max_difference = 0;
	for (j; j < 450; j++){
		float glibc_sin = bf16_to_fp32(fp32_to_bf16(sinf(bf16_to_fp32(i))));
		float payne = bf16_to_fp32(bf16_sin(i, &record_k));
		difference = fabs(payne-glibc_sin);
		if (difference > 0) {
			if (difference > max_difference) max_difference = difference;
			error_count++;
		}
		fprintf(f1, "%f\n",fabs(payne-glibc_sin));
		i.bits--;
	}
	printf("Test range [0, pi/2]:\nMax difference = %f, Total %d numbers difference\n\n", max_difference, error_count);

	// x larger than pi/2, need range reduction
	i = (bf16_t){.bits = 0b0011111111001010}; // Initial
	j = 0;
	for (j; j < 16310; j++){
		float glibc_sin = bf16_to_fp32(fp32_to_bf16(sinf(bf16_to_fp32(i))));
		float payne = bf16_to_fp32(bf16_sin(i, &record_k));
		difference = fabs(payne-glibc_sin);
		if (difference > 0) {
			if (difference > max_difference) max_difference = difference;
			error_count++;
		}
		fprintf(f2, "%f\n",fabs(payne-glibc_sin));
		i.bits++;
	}
	printf("Test range [pi/2, inf):\nMax difference = %f, Total %d numbers difference\n\n", max_difference, error_count);

	// x is negative, need range reduction
	i = (bf16_t){.bits = 0x8080}; // Initial value, smallest negative normal BF16
	j = 0;
	for (j; j < 32512; j++){
		float glibc_sin = bf16_to_fp32(fp32_to_bf16(sinf(bf16_to_fp32(i))));
		float payne = bf16_to_fp32(bf16_sin(i, &record_k));
		difference = fabs(payne-glibc_sin);
		if (difference > 0) {
			if (difference > max_difference) max_difference = difference;
			error_count++;
		}
		fprintf(f3, "%f\n",fabs(payne-glibc_sin));
		i.bits++;
	}
	printf("Test range [-smallest, -inf):\nMax difference = %f,"
	       "Total %d numbers difference\n\n", max_difference, error_count);

	fclose(f1);
	fclose(f2);
	fclose(f3);
	return 0;
}

