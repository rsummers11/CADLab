/**
 * dropc dev implementation
 *
 */

#include "dropc/dropc_bit_dev.hpp"
#include <cassert>
#include <curand_kernel.h>
#include <algorithm>

#include "dropc/cuda_common.hpp"

#define FCDROPC_BLK_SIZE 16

// texutre reference
texture<unsigned int,cudaTextureType2D,cudaReadModeElementType> texMaskWeights;
texture<float,cudaTextureType1D,cudaReadModeElementType> texMaskBiases;

// decode texMaskWeights 
__device__ __inline__
float isBitOn(
        size_t row,   ///< [in] row index 
        size_t col    ///< [in] col index 
        ){
    size_t xidx = row;
    size_t unit_size = ( sizeof(unsigned int)*8 );
    size_t yidx = col / unit_size;
    size_t offset = col % unit_size;
    unsigned int value = tex2D( texMaskWeights, xidx, yidx );
    unsigned int result = value &  ( 1 << offset );
    if( result > 0 )
        return 1.0f;
    else
        return 0.0f;
}

// FCDrop connection fprop kernel
__global__ void kFCDropC_bit_fprop(
        const float* x,      ///<[in]  input matrix x, col major, numData x inDim
        const float* w,      ///<[in]  weight matrix w, col major, inDim x outDim
        const float* b,      ///<[in]  bias matrix, row major, 1 x outDim
        int m,               ///<[in]  output dimension
        int n,               ///<[in]  input dimension
        int d,               ///<[in]  number of data in this batch
        float* y             ///<[in,out] target matrix y, col major, dataDim x outDim
        ){
    // bx,by,tx,ty
    int bx = blockIdx.x * FCDROPC_BLK_SIZE;
    int by = blockIdx.y * FCDROPC_BLK_SIZE;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // define shared memory
    __shared__ float sA[FCDROPC_BLK_SIZE][FCDROPC_BLK_SIZE];
    __shared__ float sB[FCDROPC_BLK_SIZE][FCDROPC_BLK_SIZE];

    float c = 0;
    if( (bx+tx) < m && (by + ty) < d ) {
        // get old y value 
        c = y[ (bx+tx)*d + (by+ty) ];
    }

    //loop over cols of x and rows of w
    for( int i = 0; i < n; i += FCDROPC_BLK_SIZE ) {
        // load value from x, w into shared memory and sync
        if( (i+tx) < n && (by+ty) < d ) 
            sA[ty][tx] = x[(i+tx)*d + (by+ty)];
        else
            sA[ty][tx] = 0.0f;
        if( (i+ty) < n && (bx+tx) < m )
            sB[ty][tx] = w[(tx+bx)*n + (i+ty)];
        else
            sB[ty][tx] = 0.0f;
        __syncthreads();

        // inc c value
        if( (bx+tx) < m && (by + ty) < d ) {
#pragma unroll
            for( int j = 0; j < FCDROPC_BLK_SIZE; j++ ) {
                float maskW = 0.0f;
                if( (i+j) < n ) {
                    // get m row: (i+j), col: (by+ty)^th matrix of col bx+tx
                    // size_t maskW_index = size_t(m)*n*(by+ty)+(bx+tx)*n + (i+j);
                    size_t col = size_t(m)*(by+ty)+(bx+tx);
                    maskW = isBitOn( (i+j), col );
                }
                c += sA[ty][j] * sB[j][tx] * maskW;
            }
        }
        __syncthreads(); 
    }

    if( (bx+tx) < m && (by + ty) < d ) {
        float maskB = tex1Dfetch( texMaskBiases, (tx+bx)*d+(ty+by) );
        // inc c value by bias
        c += b[tx+bx] * maskB;
        y[ (bx+tx)*d + (by+ty) ] = c;
    }
}

// FCDrop connection bprop act kernel
__global__ void kFCDropC_bit_bpropa(
        const float*  v,         ///<[in]  bprop act from previous layer, col major,numData x outDim
        const float*  w,         ///<[in]  weight matrix w, col major, inDim x outDim
        int m,                   ///<[in]  output dimension
        int n,                   ///<[in]  input dimension
        int d,                   ///<[in]  number of data in this batch
        float scale_g,           ///<[in]  input gradient scale
        float* da,               ///<[in,out] d-active, col major, numData x inDim              
        float scale_da           ///<[in]  da scale
        ){
    // bx,by,tx,ty
    int bx = blockIdx.x * FCDROPC_BLK_SIZE;
    int by = blockIdx.y * FCDROPC_BLK_SIZE;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // define shared memory
    __shared__ float sA[FCDROPC_BLK_SIZE+1][FCDROPC_BLK_SIZE+1];
    __shared__ float sB[FCDROPC_BLK_SIZE+1][FCDROPC_BLK_SIZE+1];

    float prev_da = 0;
    if( (bx+tx) < n && (by+ty)<d ) {
        // get old value
        prev_da = da[ (bx+tx)*d + (by+ty) ];
    }

    float c = 0;
    //loop over cols of v and rows of w^T(cols of w)
    for( int i = 0; i < m; i+= FCDROPC_BLK_SIZE ) {
        // load value from v and wt into shared memory and sync
        if( (i+tx) < m && (by+ty) < d ) 
            sA[ty][tx] = v[ (i+tx)*d + (by+ty) ];
        else
            sA[ty][tx] = 0.0f;
        if( (i+ty) < m && (tx+bx) < n ){
            // wt row: i+ty col: tx+bx
            sB[ty][tx] = w[ (i+ty)*n + (tx+bx) ];
        }
        else
            sB[ty][tx] = 0.0f;
        __syncthreads();

        // inc c value
        if( (bx+tx) < n && (by+ty)<d ) {
#pragma unroll
            for( int j = 0; j < FCDROPC_BLK_SIZE; j++ ) {
                float maskW = 0.0f;
                if( (i+j) < m ) {
                    //size_t maskW_index = size_t(m)*n*(ty+by)+(i+j)*n+(tx+bx);
                    size_t col = size_t(m)*(ty+by)+(i+j);
                    maskW = isBitOn( (tx+bx), col );
                }
                c += sA[ty][j] * sB[j][tx] * maskW;
            }
        }
        __syncthreads();
    }

    // set output data
    if( (bx+tx) < n && (by+ty)<d ) {
        da[ (bx+tx)*d + (by+ty) ] = prev_da * scale_da + scale_g * c;
    }
}

//FCDrop connection bprop weights kernel
__global__ void kFCDropC_bit_bpropw(
        const float* a,            ///<[in] prev activation matrix, col major, numData x inDim
        const float* v,            ///<[in] gradient matrix, col major, numData x outDim
        int m,                     ///<[in]  output dimension              
        int n,                     ///<[in]  input dimension
        int d,                     ///<[in]  number of data in this batch
        float scale_g,             ///<[in] inc scale
        float* dw,                 ///<[in,out] w gradient, col major, inDim x outDim
        float scale_dw             ///<[in] gradient scale
        ){
    // bx,by,tx,ty
    int bx = blockIdx.x * FCDROPC_BLK_SIZE;
    int by = blockIdx.y * FCDROPC_BLK_SIZE;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // define shared memory
    __shared__ float sA[FCDROPC_BLK_SIZE+1][FCDROPC_BLK_SIZE+1];
    __shared__ float sB[FCDROPC_BLK_SIZE+1][FCDROPC_BLK_SIZE+1];

    float prev_dw = 0;
    if( (bx+tx) < m && (by+ty) < n ) {
        // get the old value
        prev_dw = dw[ (bx+tx)*n + (by+ty) ];
    }

    float c = 0;
    // loop over cols of a^T(rows of a) and rows of v
    for( int i = 0; i < d; i += FCDROPC_BLK_SIZE ) {
        // load value from at and v into shared memory and sync
        if( (ty+by) < n && (i+tx) < d ) 
            sA[ty][tx] = a[ (by+ty)*d + (i+tx) ];
        else
            sA[ty][tx] = 0.0f;

        if( (tx+bx) < m && (i+ty) < d )
            sB[ty][tx] = v[ (bx+tx)*d + (i+ty) ];
        else
            sB[ty][tx] = 0.0f;
        __syncthreads();

        // inc c value
        if( (bx+tx) < m && (by+ty) < n ) {
#pragma unroll
            for( int j = 0; j < FCDROPC_BLK_SIZE; j++ ) {
                float maskW = 0.0f;
                if( (i+j) < d ) {
                    //size_t maskW_index = size_t(m)*n*(i+j)+(bx+tx)*n+(by+ty);
                    size_t col = size_t(m)*(i+j)+(bx+tx);
                    maskW = isBitOn( (by+ty), col );
                }
                c += sA[ty][j] * sB[j][tx] * maskW;
            }
        }
        __syncthreads();
    }

    // set output data
    if( (bx+tx) < m && (by+ty) < n ) {
        // get the old value
        dw[ (bx+tx)*n + (by+ty) ] = prev_dw * scale_dw + scale_g * c;
    }
}

//----------------------------------------------------
//           dev code: training
//----------------------------------------------------
void computeFCDropC_bit_fprop_d( 
        const float*  x,          ///<[in]  input matrix x, col major, numData x inDim
        const float*  w,          ///<[in]  weight matrix w, col major, inDim x outDim
        const float*  b,          ///<[in]  bias matrix, row major, 1 x outDim
        int outDim,               ///<[in]  output dimension
        int inDim,                ///<[in]  input dimension
        int numData,              ///<[in]  number of data in this batch
        const MaskWeights& mw,    ///<[in]  mask weight object
        const float * mb,         ///<[in]  maskBiases, col major, dataDim x outDim          
        float * y                 ///<[in,out] target matrix y, col major, dataDim x outDim
        ){
    // bind texture for maskWeights 
    cudaChannelFormatDesc channelDescUint = cudaCreateChannelDesc<unsigned int>();
    size_t offset_w;
    checkCuda( cudaBindTexture2D( &offset_w, &texMaskWeights, mw.data(), 
                &channelDescUint, 
                mw.get_width(), mw.get_height(),
                sizeof(unsigned int) * mw.stride()
                ) );
    assert( offset_w == 0 );
    // set clamp,point mode
    texMaskWeights.addressMode[0] = cudaAddressModeClamp;
    texMaskWeights.addressMode[1] = cudaAddressModeClamp;
    texMaskWeights.filterMode = cudaFilterModePoint;
    texMaskWeights.normalized= false;

    // bind texture for maskBiases
    cudaChannelFormatDesc channelDescFloat = cudaCreateChannelDesc<float>();
    size_t offset_b;
    checkCuda( cudaBindTexture( &offset_b, &texMaskBiases, mb, 
                &channelDescFloat, sizeof(float) * numData * outDim ) );
    assert( offset_b == 0 );
    // set clamp,point mode
    texMaskBiases.addressMode[0] = cudaAddressModeClamp;
    texMaskBiases.filterMode = cudaFilterModePoint;
    texMaskBiases.normalized= false;

    // define block/thread info
    dim3 threads( FCDROPC_BLK_SIZE, FCDROPC_BLK_SIZE );
    dim3 blocks( divup(outDim,FCDROPC_BLK_SIZE), divup(numData, FCDROPC_BLK_SIZE) );
    // invoke kernel
    kFCDropC_bit_fprop<<<blocks,threads>>>( x, w, b,
            outDim, inDim, numData, 
            y );
    // check error
    checkLastCudaError();

    // unbind texture
    checkCuda( cudaUnbindTexture( &texMaskWeights ) );
    checkCuda( cudaUnbindTexture( &texMaskBiases ) );
}


void computeFCDropC_bit_bpropActs_d(
        const float*  v,         ///<[in]  bprop act from previous layer, col major,numData x outDim
        const float*  w,         ///<[in]  weight matrix w, col major, inDim x outDim
        int outDim,              ///<[in]  output dimension
        int inDim,               ///<[in]  input dimension
        int numData,             ///<[in]  number of data in this batch
        float scale_g,           ///<[in]  input gradient scale
        const MaskWeights& mw,    ///<[in]  mask weight object
        float* da,               ///<[in,out] d-active, col major, numData x inDim              
        float scale_da           ///<[in]  da scale
        ){
    // bind texture for maskWeights 
    cudaChannelFormatDesc channelDescUint = cudaCreateChannelDesc<unsigned int>();
    size_t offset_w;
    checkCuda( cudaBindTexture2D( &offset_w, &texMaskWeights, mw.data(), 
                &channelDescUint, 
                mw.get_width(), mw.get_height(),
                sizeof(unsigned int) * mw.stride()
                ) );
    assert( offset_w == 0 );
    // set clamp,point mode
    texMaskWeights.addressMode[0] = cudaAddressModeClamp;
    texMaskWeights.filterMode = cudaFilterModePoint;
    texMaskWeights.normalized= false;

    // define block/thread info
    dim3 threads( FCDROPC_BLK_SIZE, FCDROPC_BLK_SIZE );
    dim3 blocks( divup(inDim,FCDROPC_BLK_SIZE), divup(numData, FCDROPC_BLK_SIZE) );

    // invoke kernel
    kFCDropC_bit_bpropa<<<blocks,threads>>>( v, w, 
            outDim, inDim, numData,
            scale_g,
            da, scale_da );

    // check error
    checkLastCudaError();

    // unbind texture
    checkCuda( cudaUnbindTexture( &texMaskWeights ) );
    checkCuda( cudaUnbindTexture( &texMaskBiases ) );
}

void computeFCDropC_bit_bpropWeights_d(
        const float* a,            ///<[in] prev activation matrix, col major, numData x inDim
        const float* v,            ///<[in] gradient matrix, col major, numData x outDim
        int outDim,                ///<[in]  output dimension              
        int inDim,                 ///<[in]  input dimension
        int numData,               ///<[in]  number of data in this batch
        float scale_g,             ///<[in] inc scale
        const MaskWeights& mw,    ///<[in]  mask weight object
        float* dw,                 ///<[in,out] w gradient, col major, inDim x outDim
        float scale_dw             ///<[in] gradient scale
        ){
    // bind texture for maskWeights 
    cudaChannelFormatDesc channelDescUint = cudaCreateChannelDesc<unsigned int>();
    size_t offset_w;
    checkCuda( cudaBindTexture2D( &offset_w, &texMaskWeights, mw.data(), 
                &channelDescUint, 
                mw.get_width(), mw.get_height(),
                sizeof(unsigned int) * mw.stride()
                ) );
    assert( offset_w == 0 );
    // set clamp,point mode
    texMaskWeights.addressMode[0] = cudaAddressModeClamp;
    texMaskWeights.filterMode = cudaFilterModePoint;
    texMaskWeights.normalized= false;

    // define block/thread info
    dim3 threads( FCDROPC_BLK_SIZE, FCDROPC_BLK_SIZE );
    dim3 blocks( divup(outDim,FCDROPC_BLK_SIZE), divup(inDim, FCDROPC_BLK_SIZE) );

    // invoke kernel
    kFCDropC_bit_bpropw<<<blocks,threads>>>( a, v, 
            outDim, inDim, numData,
            scale_g,
            dw, scale_dw );

    // check error
    checkLastCudaError();

    // unbind texture
    checkCuda( cudaUnbindTexture( &texMaskWeights ) );
    checkCuda( cudaUnbindTexture( &texMaskBiases ) );
}

//----------------------------------------------------
//         MaskWeights related kernels
//----------------------------------------------------

#define FCDROPC_RND_GRID_SIZE    8
#define FCDROPC_RND_BLK_SIZE     8

__global__ void kInitRandomSeed(
        curandState *state,      ///< [in,out] allocated random state
        unsigned long long seed  ///< [in] random seed
        ) {
    const unsigned int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    curand_init(seed, tidx, 0, &state[tidx]);
}

__global__ void kRandomizeMaskWeight( 
        curandState* state,  ///<[in,out] curand state
        float onProb,        ///<[in] on probability
        unsigned int* mw,    ///<[out] mw, col major, height x width
        int width,           ///<[in] matrix width
        int height,          ///<[in] matrix height
        size_t stride        ///<[in] matrix col stride
        ){
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x * blockDim.x;
    int by = blockIdx.y * blockDim.y;
    int sx = blockDim.x * gridDim.x;
    int sy = blockDim.x * gridDim.y;
    // get random state
    int ridx = (by+ty) * FCDROPC_RND_BLK_SIZE * FCDROPC_RND_GRID_SIZE +(bx+tx);
    //int ridx = (ty) * FCDROPC_RND_BLK_SIZE +(tx);
    curandState localState = state[ridx];

    // randomize data
    for( int x = 0; x < width; x+= sx )
        for( int y = 0; y < height; y += sy ) {
            if( (y+ty+by) < height && (x+tx+bx) < width ) {
                size_t data_idx = (y+ty+by)*stride + (x+tx+bx);
                unsigned int result = 0;
#pragma unroll
                for( size_t i = 0; i < sizeof(unsigned int)*8; i++ ){
                    //mw[data_idx] = curand( &localState );
                    float v = curand_uniform( &localState );
                    // set to one
                    if( v < onProb ) {
                        result |= ( 1 << i );
                    }
                }
                mw[data_idx] = result;
            }
        }

    // store back
    state[ridx] = localState;
}


//----------------------------------------------------
//         MaskWeights class definition 
//----------------------------------------------------
// default constr
MaskWeights::MaskWeights( float onProb, unsigned long long seed ) : seed_(seed),
    outDim_(0), inDim_(0), numData_(0),
    stride_(0), data_(NULL), devStates_(NULL) , onProb_(onProb)
{
    initRandomSeed();
}

MaskWeights::~MaskWeights() {
    checkCuda( cudaFree( devStates_ ) );
    checkCuda( cudaFree( data_ ) );
}

void MaskWeights::initRandomSeed() {
    if( devStates_ != NULL )
        return;
    // init reandom seed
    size_t num = FCDROPC_RND_BLK_SIZE * FCDROPC_RND_GRID_SIZE;
    checkCuda( cudaMalloc( &devStates_, num * num * sizeof(curandState) ) );
    // invoke kernel
    dim3 thread( FCDROPC_RND_BLK_SIZE * FCDROPC_RND_BLK_SIZE );
    dim3 block( FCDROPC_RND_GRID_SIZE * FCDROPC_RND_GRID_SIZE );
    kInitRandomSeed<<< block, thread >>>( devStates_, seed_ ); 
    // check error
    checkLastCudaError();
}

size_t MaskWeights::get_height() const {
    return divup( outDim_ * numData_, sizeof(unsigned int) * 8 );
}

size_t MaskWeights::get_width() const {
    return inDim_;
}

/** set onProb */
void MaskWeights::set_onProb( float onProb ) {
    assert( onProb >=0 && onProb <= 1 );
    onProb_ = onProb;
}

//resize 
void MaskWeights::resize(
        int outDim,            ///<[in] new outDim
        int inDim,             ///<[in] new inDim
        int numData            ///<[in] new numData
        ){
    // simple retur if size does not change
    if( outDim_ == outDim &&
            inDim_ == inDim &&
            numData_ == numData ) {
        return;
    }
    // release memory if necessary
    if( outDim_ > 0 && inDim_ > 0 && numData_ > 0 ) {
        checkCuda( cudaFree( data_ ) );
    }
    assert( outDim >= 0 && inDim >=0 && numData >= 0 );
    outDim_ = outDim;
    inDim_ = inDim;
    numData_ = numData;
    int width = get_width();
    int height = get_height();
    checkCuda( cudaMallocPitch( &data_, &stride_ , 
                width * sizeof(unsigned int), height ) );
    stride_ /= sizeof( unsigned int );
}

void MaskWeights::randomize(){
    int width = get_width();
    int height = get_height();
    // invoke kernel
    dim3 thread( FCDROPC_RND_BLK_SIZE, FCDROPC_RND_BLK_SIZE );
    int bx = min( divup( width, FCDROPC_RND_BLK_SIZE ), FCDROPC_RND_GRID_SIZE); 
    int by = min( divup( height, FCDROPC_RND_BLK_SIZE ), FCDROPC_RND_GRID_SIZE);
    dim3 block( bx, by );

    kRandomizeMaskWeight<<< block, thread >>>( devStates_, onProb_, data_,
            width, height, stride_ );
    // check error
    checkLastCudaError();
}

//----------------------------------------------
//           dev code: inference
//----------------------------------------------
#define FCDROPC_RND_GRID_1D_SIZE   32
#define FCDROPC_RND_BLK_1D_SIZE    128

struct ReluNeuron {
    __device__ __inline__
        float operator()( float x ) {
            return  x > 0 ? x : 0;
        }
};

template<class NeuronResponse>
__global__ void kFCDropC_Inference( 
        const float*  mu,       ///<[in]  mean matrix, col major, dataDim x outDim
        const float*  var,      ///<[in]  var matrix,  col major, dataDim x outDim
        size_t n,               ///<[in]  vector length
        NeuronResponse nr,      ///<[in] functor compute neuron response
        int numSamples,         ///<[in]  number of samples for mc sampling
        float * y,              ///<[in,out] target matrix y, col major, dataDim x outDim
        curandState* state      ///<[in,out] curand state
        ){
    int tx = threadIdx.x;
    int bx = blockIdx.x * blockDim.x;
    int stride = FCDROPC_RND_BLK_1D_SIZE * FCDROPC_RND_GRID_1D_SIZE;
    // get current state
    curandState localState = state[bx+tx];
    for( int sx = 0; sx < n; sx += stride ) {
        int cx = sx + bx + tx;
        if( cx < n ) {
            float mux = mu[cx];
            float sigmax = sqrtf(var[cx]);
            float rz = 0.0f;
            for( int i = 0; i < numSamples; i++ ) {
                float xi = curand_normal( &localState );
                xi *= sigmax;
                xi += mux;
                // apply neuron
                rz += (NeuronResponse()(xi))/numSamples;
            }
            y[cx]=rz;
        }
    }
    // store back random state
    state[bx+tx]=localState;
}

void computeFCDropC_bit_inference_d(
        const float*  mu,       ///<[in]  mean matrix, col major, dataDim x outDim
        const float*  var,      ///<[in]  var matrix,  col major, dataDim x outDim
        size_t n,               ///<[in]  vector length
        int numSamples,         ///<[in]  number of samples for mc sampling
        float * y               ///<[in,out] target matrix y, col major, dataDim x outDim
        ){
    // init random seed
    unsigned long long seed = 0;
    curandState* devStates; 
    size_t num = FCDROPC_RND_BLK_1D_SIZE * FCDROPC_RND_GRID_1D_SIZE;
    checkCuda( cudaMalloc( &devStates, num * sizeof(curandState) ) );
    dim3 thread, block;
    // invoke init random kernel
    dim3 threads( FCDROPC_RND_BLK_1D_SIZE );
    dim3 blocks( FCDROPC_RND_GRID_1D_SIZE );
    kInitRandomSeed<<< blocks, threads >>>( devStates, seed ); 
    checkLastCudaError();
    // invoke random compute kernel
    size_t blks = divup( n, FCDROPC_RND_BLK_1D_SIZE );
    if( blks < FCDROPC_RND_GRID_1D_SIZE )
       blocks = dim3( blks );
    kFCDropC_Inference<<<blocks, threads>>>( 
            mu, var, n, ReluNeuron(),
            numSamples, y, devStates );
    checkLastCudaError();
    // clean up
    checkCuda( cudaFree( devStates ) );
}
