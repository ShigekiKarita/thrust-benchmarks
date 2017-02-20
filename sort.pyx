cdef extern from "thrust.cuh" namespace "my_thrust":
    # void cu_stable_sort "my_thrust::stable_sort" ()
    void cu_stable_sort_batch_vector "my_thrust::stable_sort_batch_vector" ()
    void cu_stable_sort_batch_nested "my_thrust::stable_sort_batch_nested" ()

# def stable_sort():
#     cu_stable_sort()

def stable_sort_vector():
    cu_stable_sort_batch_vector()

def stable_sort_nested():
    cu_stable_sort_batch_nested()
