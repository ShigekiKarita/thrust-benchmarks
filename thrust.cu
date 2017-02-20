#include "thrust.cuh"

#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include <thrust/sort.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>

#include <thrust/sequence.h>
#include <thrust/for_each.h>

#include "mod_range.cuh"

namespace my_thrust {
  // void my_thrust::stable_sort() {
  //     thrust::device_ptr<float> d_ptr = thrust::device_malloc<float>(3);

  //     thrust::device_ptr<float> first = d_ptr;
  //     thrust::device_ptr<float> last  = d_ptr + 3;

  //     d_ptr[0] = 3.0; d_ptr[1] = 2.0; d_ptr[2] = 1.0;
  //     thrust::stable_sort(first, last);

  //     std::cout << d_ptr[0] << ", " << d_ptr[1] << ", " << d_ptr[2] << std::endl;

  //     thrust::device_free(d_ptr);
  // }


  static const int NSORTS = 16000;
  static const int DSIZE = 1000;

  struct Mod {
    int d_;
    int p_;
    Mod(int d) : d_(d) {}
    int operator()() {
      return p_++ / d_;
    }
  };

  thrust::device_vector<int> gen_rand() {
    thrust::host_vector<int> h_data(DSIZE*NSORTS);
    thrust::generate(h_data.begin(), h_data.end(), rand);
    thrust::device_vector<int> d_data = h_data;
    return d_data;
  }

  bool validate(const thrust::device_vector<int> &d1, const thrust::device_vector<int> &d2){
    return thrust::equal(d1.cbegin(), d1.cend(), d2.cbegin());
  }

  void print(const thrust::device_vector<int>& result) {
    std::cout << result[0] << ", " << result[1] << ", " << result[2] << " ... ";
    std::cout << result[DSIZE-3] << ", " << result[DSIZE-2] << ", " << result[DSIZE-1] << std::endl;
    int c = (NSORTS - 1) * DSIZE;
    std::cout << result[c+0] << ", " << result[c+1] << ", " << result[c+2] << " ... ";
    std::cout << result[c+DSIZE-3] << ", " << result[c+DSIZE-2] << ", " << result[c+DSIZE-1] << std::endl;
  }

  template <class Proc>
  void benchmark(Proc proc) {
    thrust::device_vector<int> d_vec = gen_rand();
    auto expect = d_vec;

    for (int i = 0; i < NSORTS; i++) {
      thrust::sort(expect.begin() + (i*DSIZE), expect.begin() + ((i+1)*DSIZE));
    }

    // https://ivanlife.wordpress.com/2011/05/09/time-cuda/
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    proc(d_vec);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float time;
    cudaEventElapsedTime(&time, start, stop);

    print(d_vec);
    printf ("Time for the kernel: %f ms\n", time);
    if (validate(d_vec, expect)) {
      printf("OK!\n");
    } else {
      printf("failed!\n");
    }
  }

  void stable_sort_batch_vector() {
    // ??? count_iterator ???
    benchmark([=](thrust::device_vector<int>& d_vec) {
        // thrust::host_vector<int> h_segments(DSIZE*NSORTS);
        // thrust::generate(h_segments.begin(), h_segments.end(), Mod(DSIZE));
        // thrust::device_vector<int> d_segments = h_segments;
        typedef thrust::device_vector<int>::iterator Iterator;
        mod_range<Iterator> d_segments(d_vec.begin(), d_vec.end(), DSIZE);

        thrust::stable_sort_by_key(d_vec.begin(), d_vec.end(), d_segments.begin());
        // thrust::stable_sort_by_key(d_segments.begin(), d_segments.end(), d_vec.begin());
      });
  }



  struct SortFunctor
  {
    thrust::device_ptr<int> data;
    int dsize;
    __host__ __device__
    void operator()(int start_idx)
    {
      thrust::sort(thrust::device, data+(dsize*start_idx), data+(dsize*(start_idx+1)));
    }
  };

  void stable_sort_batch_nested() {
    benchmark([=](thrust::device_vector<int>& d_vec) {
        cudaDeviceSetLimit(cudaLimitMallocHeapSize, (16*DSIZE*NSORTS));
        thrust::device_vector<int> d_result3 = gen_rand();
        SortFunctor f = {d_result3.data(), DSIZE};
        thrust::device_vector<int> idxs(NSORTS);
        thrust::sequence(idxs.begin(), idxs.end());
        thrust::for_each(idxs.begin(), idxs.end(), f);
      });
  }

} // namespace
