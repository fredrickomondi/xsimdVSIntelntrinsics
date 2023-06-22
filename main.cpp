#include <iostream>
#include <xsimd/xsimd.hpp>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <cstddef>
using namespace std;

#define DATA_SIZE 10000


float generate_random_float()
{
	return rand() / (float)RAND_MAX;
}


inline vector <float> compute_fma_with_standard_c_lib(const vector<float>& v)
{

	vector<float> output(DATA_SIZE, 0.0);

	for (int i = 0; i < v.size(); i++) {

		output[i] = ((v[i] * v[i]) + v[i]);

	}

	return output;
}


inline vector<float> compute_fma_with_xsimd_wrapper(const vector<float>& v)
{

	vector<float> output(DATA_SIZE, 0.0);
	using b_type = xsimd::batch<float, xsimd::avx2>;
	size_t inc = b_type::size;
	size_t size = output.size();

	for (size_t i = 0; i < size; i += inc)
	{
		b_type a = b_type::load_aligned(&v[i]);
		b_type b = b_type::load_aligned(&v[i]);
		b_type c = b_type::load_aligned(&v[i]);
		auto res = xsimd::fma(a, b, c);
		res.store_aligned(&output[i]);

	}

	return output;

}




bool a_is_close_to_b(float a, float b)
{
	constexpr float delta = 0.000001f;

	if (isnan(a) || isnan(b))
		return false;

	return (a - b) <= delta;
}

inline vector<float> compute_fma_with_native_instrinsics(const vector<float>& v)
{
	vector<float> output(DATA_SIZE, 0.0);

	for (int i = 0; i < v.size(); i += 8)
	{

		__m256 a, b, c;

		a = _mm256_load_ps(&v[i]);
		b = _mm256_load_ps(&v[i]);
		c = _mm256_load_ps(&v[i]);

		__m256 res = _mm256_fmadd_ps(a, b, c);
		_mm256_store_ps(&output[i], res);

	}

	return output;
}


vector<float> populate_random_vector()
{
	vector<float> res;
	for (int i = 0; i < DATA_SIZE; i++)
	{
		res.push_back(generate_random_float());

	}

	return res;
}


void check_best_supported_architecture()
{
	cout << "Best architecture selected by XSIMD is :: " << xsimd::best_arch::name() << endl;


}

void compute_elapsed_time(string operation, const vector<float> vector_of_random_floats, vector<float>& results, vector<float>(*operation_function)(const vector<float>&))
{
	auto begin_native = chrono::steady_clock::now();
	results = operation_function(vector_of_random_floats);
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

	check_best_supported_architecture();
	vector <float> vector_of_random_floats = populate_random_vector();
	vector<float> vector_of_fma_results_native;
	vector<float>  vector_of_fma_results_standard;
	vector<float>  vector_of_fma_results_native_sse;
	vector<float>  vector_of_fma_results_xsimd;


	cout << "***************************************************BEGIN**********************************" << endl;

	compute_elapsed_time("FMA operation using native Intel AVX2 Instrinsic", vector_of_random_floats, vector_of_fma_results_native, &compute_fma_with_native_instrinsics);
	compute_elapsed_time("FMA operation using XSIMD wrapper for AVX2", vector_of_random_floats, vector_of_fma_results_xsimd, &compute_fma_with_xsimd_wrapper);
	compute_elapsed_time("Scalar FMA operation using standard libs", vector_of_random_floats, vector_of_fma_results_standard, &compute_fma_with_standard_c_lib);


	cout << "***************************************************END**********************************" << endl;
	
	auto count = 0;
	auto missmatch = 0;

	for (auto item : vector_of_fma_results_native) {



		if (!a_is_close_to_b(item, vector_of_fma_results_xsimd[count]) && a_is_close_to_b(vector_of_fma_results_xsimd[count], vector_of_fma_results_standard[count])) {



			cout << "missmatch " << item << " --vs-- " << vector_of_fma_results_standard[count] << " --vs-- " << vector_of_fma_results_xsimd[count] << endl;
			missmatch++;
			count++;
			continue;
		}


		count++;
	}


	cout << "total missmatched result(s) is " << missmatch << " out of " << count << endl;

	return 0;
}


