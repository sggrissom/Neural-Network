/* ========================================================================
   File: NeuralNetwork.cpp
   Date: 2015-12-15
   Creator: Steven Grissom
   ======================================================================== */


#define BETA 0.3f
#define ALPHA 0.1f
#define EPSILON 0.0001f
#define MAX_ITERATIONS 10
#define TEST_ITERATIONS 1000

#define FLAT 1
#define UNROLLED 0

#define IRIS 0
#define DIGITS 1
#define XOR4 0
#define XOR2 0

#if IRIS
#define LAYERSIZES {4,100,100,2}
#define FILENAME "..\\src\\iris.data"
#include "load.cpp"
#endif

#if DIGITS
#define LAYERSIZES {256,100,10}
#define FILENAME "..\\src\\digits.data"
#include "load.cpp"
#endif

#if XOR4
#define LAYERSIZES {4,10,1}
global r32 TrainingData[] = {
    0,0,0,0,0,
    0,0,0,1,1,
    0,0,1,0,1,
    0,0,1,1,0,
    0,1,0,0,1,
    0,1,0,1,0,
    0,1,1,0,0,
    0,1,1,1,1,
    1,0,0,0,1,
    1,0,0,1,0,
    1,0,1,0,0,
    1,0,1,1,1,
    1,1,0,0,0,
    1,1,0,1,1,
    1,1,1,0,1,
    1,1,1,1,0,
};
#endif

#if XOR2
#define LAYERSIZES {2,5,5,1}
global r32 TrainingData[] = {
    0,0,0,
    0,1,1,
    1,0,1,
    1,1,0,
};
#endif
    
struct neural_network
{
    u32 LayerCount;
    u32 InputCount;
    u32 OutputCount;
    r32 Beta;
    r32 Alpha;
    r32 Epsilon;
    u32 *LayerSizes;

    r32 *Data;
    r32 *Delta;
    r32 *Weights;
    r32 *WeightDelta;

    u32 *DataRowPointer;
    u32 *WeightsRowPointer;
};

internal void
InitializeNetwork(neural_network *NeuralNetwork)
{
	r32 *data;
	r32 *delta;
	r32 *weight;
	r32 *prevDwt;
    
	u32 numl = NeuralNetwork->LayerCount;
    u32 *lsize = NeuralNetwork->LayerSizes;

	u32 numn = 0;
	u32 *rowptr_od = (u32 *)malloc(numl*sizeof(u32));
	for(u32 i=0;
        i<numl;
        ++i)
	{
		rowptr_od[i] = numn;
		numn += lsize[i];
	}
	rowptr_od[numl] = numn;

	// Allocate memory for out, delta
	data = (r32 *)malloc(numn * sizeof(r32));
	delta = (r32 *)malloc(numn * sizeof(r32));

	// Allocate memory for weights, prevDwt
	u32 numw = 0;
    u32 *rowptr_w = (u32 *)malloc((numl+1)*sizeof(u32));
	rowptr_w[0] = 0;
	for(u32 i=1; i<numl; i++)
	{
		rowptr_w[i] = numw;
		numw += lsize[i]*(lsize[i-1]+1);
	}
	weight = (r32 *)malloc(numw * sizeof(r32));
	prevDwt = (r32 *)malloc(numw * sizeof(r32));

	// Seed and assign random weights; set prevDwt to 0 for first iter
	for(u32 i=1;i<numw;i++)
	{
		weight[i] = (r32)(rand())/(RAND_MAX/2) - 1;//32767
		prevDwt[i] = 0.0f;
	}

    NeuralNetwork->Data = data;
    NeuralNetwork->Delta = delta;
    NeuralNetwork->Weights = weight;
    NeuralNetwork->WeightDelta = prevDwt;
    NeuralNetwork->DataRowPointer = rowptr_od;
    NeuralNetwork->WeightsRowPointer = rowptr_w;
}
