#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/QuickPool.hpp"
#include "chai/config.hpp"
#include "chai/ManagedArray.hpp"
#include "chai/managed_ptr.hpp"
#include "RAJA/RAJA.hpp"

#include <iostream>

/************************************************************************
* Initial sections is just a bunch of helper code used for later steps
*/

#ifdef __CUDACC__
#include <cuda_runtime_api.h>
#endif

#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#endif

#if defined(__CUDACC__) || defined(__HIPCC__)
#define __test_host__   __host__
#define __test_device__ __device__
#define __test_global__ __global__
#define __test_hdev__   __host__ __device__
#define __test_gpu_active__
#else
#define __test_host__
#define __test_device__
#define __test_global__
#define __test_hdev__
#endif

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))  || defined(__HIP_DEVICE_COMPILE__)
#define __test_device_only__      
#else
#define __test_host_only__      
#endif

#define NUMBLOCKS 256
#define ASYNC false

// Uncomment this line to disable the necessary optimizations to get the problem to not crash
// #define PASS_TEST
#if defined(__HIPCC__) && defined(PASS_TEST)
#define TEST_ROCM_OPTIMIZE_OFF() _Pragma("clang optimize off")
#define TEST_ROCM_OPTIMIZE_ON() _Pragma("clang optimize on")
#else
#define TEST_ROCM_OPTIMIZE_OFF()
#define TEST_ROCM_OPTIMIZE_ON()
#endif

template<typename T>
using rview2 = RAJA::View<T, RAJA::Layout<2>>;
using rview2d = rview2<double>;

template<typename T>
__test_host__
inline
chai::ManagedArray<T> allocManagedArray(std::size_t size=0)
{

    auto& rm = umpire::ResourceManager::getInstance();
#if defined(CHAI_ENABLE_CUDA) || defined(CHAI_ENABLE_HIP)
    auto es = chai::ExecutionSpace::GPU;
#else
    auto es = chai::ExecutionSpace::CPU;
#endif

    chai::ManagedArray<T> array(size, 
    std::initializer_list<chai::ExecutionSpace>{chai::CPU
#if defined(CHAI_ENABLE_CUDA) || defined(CHAI_ENABLE_HIP)
        , chai::GPU
#endif
        },
        std::initializer_list<umpire::Allocator>{rm.getAllocator("HOST")
#if defined(CHAI_ENABLE_CUDA) || defined(CHAI_ENABLE_HIP)
        , rm.getAllocator("DEVICE")
#endif
        },
        es
    );

    return array;
}

/************************************************************************
* The Input and Output structs are just pure helper data structs to
* utilize for our input and output for each function
*/

struct Inputs {
    __test_hdev__
    Inputs(const double deltaTime_,
           const rview2d history_,
           const size_t index_) : 
           deltaTime(deltaTime_),
           history(history_),
           index(index_)
            {}

    double deltaTime{0.0};
    const rview2d history;
    const size_t index;
};

struct Outputs {

    __test_hdev__
    Outputs(double& val0_,
            double& val1_,
            double& dval1_dval2_,
            const rview2d history_update_,
            const size_t index_) :
            val0(val0_),
            val1(val1_),
            dval1_dval2(dval1_dval2_),
            history_update(history_update_),
            index(index_) {}

    double& val0;
    double& val1;
    double& dval1_dval2;

    const rview2d history_update;
    const size_t index;

    
};

/************************************************************************
 * We have some base class that we'll be leveraging to call our down-casted derived classes in later parts of our code
 * The base class might do a lot of different things but generally we have it take in some input and output structs that contain
 * references to our various variables we need to work with.
 * From our RAJA::forall loops, we'll be calling the single public function which as can see then calls in this instance some base implementation
 * as most models don't need to override the base implementation. This base implementation makes 3 fcn calls that all play a role in this error.

 * The first fcn calls the base implementation typically updates 1-2 variables in the output struct
 * The second fcn calls the derived class implementation then for the error to occur just needs to update a variable in the output struct
 * The third fcn call then just has to try and update a variable in the output struct again for things to fail

 * I'm still working out the exact interactions here that are important but it seems that just updating certain variables in at least the first 2 kernels is
 * enough to cause a general failure in the third but I'm unsure. Initial tests seem to suggest the views to be important though to causing later issues.
*/

class ModelBase {
public:
    __test_hdev__
    ModelBase() {}

    __test_hdev__
    virtual ~ModelBase() {};

   // the function that we eventually call from our chai::managed_ptr later on.
    __test_hdev__
    void evaluate_state(Inputs& inputs,
                        Outputs& outputs) {
        update_state(inputs, outputs);
    }

protected:

    __test_hdev__
    virtual void update_terms2(const Inputs& inputs,
                                 Outputs& outputs) const = 0;

   // Interactions of various functions down below causes issues
    __test_hdev__
    virtual void update_state(const Inputs &inputs,
                              Outputs &outputs) const {
        // Makes use of base class implementation
        update_terms1(inputs, outputs);
        // Makes use of the derived class implementation
        update_terms2(inputs, outputs);
        // Makes use of the base class implementation again
        update_terms3(inputs, outputs);
    }
    __test_hdev__
    virtual void update_terms3(const Inputs & /* inputs*/,
                               Outputs &outputs) const {
        // Can't find a strong correllation to which variable we access here that cause the failure
        // It ultimately looks like it just has to be something from the Output struct
        outputs.val0 = 0.0;
        // outputs.history_update(1) = outputs.val1;
        // outputs.history_update(2) = 0.0 ;
    }

TEST_ROCM_OPTIMIZE_OFF()
    __test_hdev__
    void update_terms1(const Inputs &inputs,
                       Outputs &outputs) 
                          const {
        // Originally, we were modifiying the RAJA view in the output but it turns
        // out we just have to access the input RAJA view to cause later issues or
        // at least that's what my initial tests have shown.
        outputs.val1 = inputs.deltaTime + inputs.history(inputs.index, 0);
        // outputs.history_update(0) = hist(0) + val0 * deltaTime ;
    }
TEST_ROCM_OPTIMIZE_ON()
};

class ModelDummy final : public ModelBase {
public:
    __test_hdev__
    ModelDummy() {}

private:

    // If we touch any memory from the output's struct things fail
    // However, if we make this function trivial then things work
    __test_hdev__
    virtual void update_terms2(const Inputs &,
                                 Outputs &outputs) const override {
        // merely updating any of the memory in the struct causes issues the failures
        outputs.val1 = 2.0 * outputs.history_update(outputs.index, 1) * outputs.val0 ;
        outputs.dval1_dval2 = 2.0 * outputs.history_update(outputs.index, 1);
    }
};

int main() {

    const size_t npts = 1;
    const size_t nhist = 3;


    auto hist_array = allocManagedArray<double>(npts * nhist);
    auto hist_array_updt = allocManagedArray<double>(npts * nhist);


#if defined(RAJA_ENABLE_CUDA)
    using gpu_exec_policy = RAJA::cuda_exec<NUMBLOCKS, ASYNC>;
#else
    using gpu_exec_policy = RAJA::hip_exec<NUMBLOCKS, ASYNC>;
#endif

    auto raja_range =  RAJA::RangeSegment(0, npts);

    RAJA::forall<RAJA::seq_exec>(raja_range, [=] __test_hdev__(size_t i) {
        hist_array[i * nhist + 0 ] = 1.0;
        hist_array[i * nhist + 1 ] = 1.0;
        hist_array[i * nhist + 2 ] = 0.0;
        hist_array_updt[i * nhist + 0 ] = hist_array[i * nhist + 0 ];
        hist_array_updt[i * nhist + 1 ] = hist_array[i * nhist + 1 ];
        hist_array_updt[i * nhist + 2 ] = hist_array[i * nhist + 2 ];
    });


    const rview2d hist_view(hist_array.data(chai::ExecutionSpace::GPU), npts, nhist);
    rview2d hist_view_updt(hist_array_updt.data(chai::ExecutionSpace::GPU), npts, nhist);

    auto model_ptr = chai::make_managed<ModelDummy>();

    const double delta_time = 1.0;

    RAJA::forall<gpu_exec_policy>(raja_range, [=] __test_hdev__(size_t i) {
        double val1 = 0.0;
        double val0 = 0.0;
        double dval1_dval2 = 0.0;
        Inputs inputs = Inputs(delta_time, hist_view, i);
        Outputs outputs = Outputs(val1, val0, dval1_dval2, hist_view_updt, i);
        model_ptr->evaluate_state(inputs, outputs);
    });

    model_ptr.free();

    // If this doesn't fail then 
    RAJA::forall<RAJA::seq_exec>(raja_range, [=] __test_hdev__(size_t i) {
#ifdef __test_host_only__
        std::cout << "hist_updt: " << hist_view_updt(i, 0) << " " << hist_view_updt(i, 1) << std::endl;
#endif
    });

    return 0;

}