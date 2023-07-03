#include <iostream>
#include <memory>
#include <xsimd/xsimd.hpp>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <cstddef>
using namespace std;

#define DATA_SIZE 10000000


#ifndef ALWAYS_INLINE
#if defined __GNUC__
#define ALWAYS_INLINE inline __attribute__((__always_inline__))
#elif defined _MSC_VER
#define ALWAYS_INLINE __forceinline
#else
#define ALWAYS_INLINE inline
#endif
#endif

float generate_random_float()
{
	return rand() / (float)RAND_MAX;
}

struct compute_fma_with_standard_c_lib {
ALWAYS_INLINE void operator()(int size, float *v, float *output)
{
	for (int i = 0; i < size; i++) {
		output[i] = ((v[i] * v[i]) + v[i]);
	}
}
};

struct compute_fma_with_xsimd_wrapper {
ALWAYS_INLINE void operator()(int size, float *v, float *output)
{
	using b_type = xsimd::batch<float, xsimd::avx2>;
	size_t inc = b_type::size;

	for (size_t i = 0; i < size; i += inc)
	{
		b_type a = b_type::load_aligned(&v[i]);
		b_type b = b_type::load_aligned(&v[i]);
		b_type c = b_type::load_aligned(&v[i]);
		auto res = xsimd::fma(a, b, c);
		res.store_aligned(&output[i]);
	}
}
};


bool a_is_close_to_b(float a, float b)
{
	constexpr float delta = 0.000001f;

	if (isnan(a) || isnan(b))
		return false;

	return (a - b) <= delta;
}

struct compute_fma_with_native_instrinsics {
ALWAYS_INLINE void operator()(int size, float *v, float *output)
{
	for (int i = 0; i < size; i += 8)
	{

		__m256 a, b, c;

		a = _mm256_load_ps(&v[i]);
		b = _mm256_load_ps(&v[i]);
		c = _mm256_load_ps(&v[i]);

		__m256 res = _mm256_fmadd_ps(a, b, c);
		_mm256_store_ps(&output[i], res);

	}
}
};

// for posix_memalign()
#include <stdlib.h>

#if defined __WIN32
#define MEMALIGN_ALLOC(p, a, s) ((*(p)) = _aligned_malloc((s), (a)), *(p) ? 0 : errno)
#define MEMALIGN_FREE(p) _aligned_free((p))
#else
#define MEMALIGN_ALLOC(p, a, s) posix_memalign((p), (a), (s))
#define MEMALIGN_FREE(p) free((p))
#endif

auto populate_random_array()
{
	void *ptr = nullptr;
	MEMALIGN_ALLOC(&ptr, 8 * sizeof(float), DATA_SIZE * sizeof(float));

	float *floatPtr = static_cast<float*>(ptr);

	auto deleter = [] (float*ptr){MEMALIGN_FREE(ptr);};

	std::unique_ptr<float[], decltype(deleter)> realPointer(floatPtr, deleter);

	for (int i = 0; i < DATA_SIZE; i++)
	{
		*floatPtr = generate_random_float();
		++floatPtr;
	}

	return realPointer;
}


void check_best_supported_architecture()
{
	cout << "Best architecture selected by XSIMD is :: " << xsimd::best_arch::name() << endl;


}

template<class Functor>
void compute_elapsed_time(string operation, int size, float *vector_of_random_floats, float *results)
{
	auto begin_native = chrono::steady_clock::now();
	Functor()(size, vector_of_random_floats, results);
	auto end_native = chrono::steady_clock::now();
	cout << "Elapsed time for " + operation + " is :: " << chrono::duration_cast<chrono::microseconds> (end_native - begin_native).count() << " microseconds " << std::endl;
	cout << endl;
}



void print(vector<float> res) {
	for (auto item : res) {
		cout << item << endl;
	}
}

int main() {

	const int size = DATA_SIZE;

	check_best_supported_architecture();
	auto vector_of_random_floats = populate_random_array();
	auto vector_of_fma_results_native = populate_random_array();
	auto vector_of_fma_results_standard = populate_random_array();
	auto vector_of_fma_results_native_sse = populate_random_array();
	auto vector_of_fma_results_xsimd = populate_random_array();

	cout << "***************************************************BEGIN**********************************" << endl;

	compute_elapsed_time<compute_fma_with_native_instrinsics>("FMA operation using native Intel AVX2 Instrinsic", size, vector_of_random_floats.get(), vector_of_fma_results_native.get());
	compute_elapsed_time<compute_fma_with_xsimd_wrapper>("FMA operation using XSIMD wrapper for AVX2", size, vector_of_random_floats.get(), vector_of_fma_results_xsimd.get());
	compute_elapsed_time<compute_fma_with_standard_c_lib>("Scalar FMA operation using standard libs", size, vector_of_random_floats.get(), vector_of_fma_results_standard.get());


	cout << "***************************************************END**********************************" << endl;
	
	auto count = 0;
	auto missmatch = 0;

	for (int i = 0; i < size; ++i) {



		if (!a_is_close_to_b(vector_of_fma_results_native[count], vector_of_fma_results_xsimd[count]) && a_is_close_to_b(vector_of_fma_results_xsimd[count], vector_of_fma_results_standard[count])) {



			cout << "missmatch " << vector_of_fma_results_native[count] << " --vs-- " << vector_of_fma_results_standard[count] << " --vs-- " << vector_of_fma_results_xsimd[count] << endl;
			missmatch++;
			count++;
			continue;
		}


		count++;
	}


	cout << "total missmatched result(s) is " << missmatch << " out of " << count << endl;

	return 0;
}


