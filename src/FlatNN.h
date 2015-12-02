
//NOTE(steven):Data[Layer][neuron]
//Weights[Layer][Neuron][WeightsForPrevLayerNeuron]

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>

internal r32
Sigmoid(r32 X)
{
    r32 Result = (r32)(1.0f/(1.0f + exp(-X)));
    Assert(Result < 1.0f && Result > 0);

    return Result;
}

internal void
SeedArrayRandomly(r32 *Array, u32 ArraySize)
{
    for(u32 ArrayIndex = 0;
        ArrayIndex < ArraySize;
        ++ArrayIndex)
    {
#if 1
        r32 RandomNumber = (r32)rand() / (RAND_MAX/2) - 1;
#else
        r32 RandomNumber = (r32)ArrayIndex;
#endif
        Array[ArrayIndex] = RandomNumber;
    }
}

struct flat_2d_array
{
    r32 *Array;
    u32 *RowPointer;
    u32 MemberCount;
    u32 RowCount;
};

struct neural_network
{   
    u32 InputCount;
    u32 OutputCount;
    u32 *LayerSizes;
    u32 LayerCount;
    u32 MaximumIterations;
    r32 Beta;
    r32 Alpha;
    r32 Epsilon;
    
    flat_2d_array *Data;
    flat_2d_array *Delta;
    flat_2d_array *Weights;
    flat_2d_array *PreviousWeights;
};

internal void
InitializeNeuralNetwork(neural_network *NeuralNetwork, 
                        u32 InputCount,
                        u32 OutputCount,
                        u32 *LayerSizes,
                        u32 LayerCount,
                        r32 Beta,
                        r32 Alpha,
                        r32 Epsilon,
                        u32 MaximumIterations)
{
    NeuralNetwork->InputCount = InputCount;
    NeuralNetwork->OutputCount = OutputCount;
    NeuralNetwork->LayerCount = LayerCount;
    NeuralNetwork->LayerSizes = LayerSizes;
    NeuralNetwork->Beta = Beta;
    NeuralNetwork->Alpha = Alpha;
    NeuralNetwork->Epsilon = Epsilon;
    NeuralNetwork->MaximumIterations = MaximumIterations;

    flat_2d_array *Data = (flat_2d_array *)malloc(sizeof(flat_2d_array));
    flat_2d_array *Delta = (flat_2d_array *)malloc(sizeof(flat_2d_array));
    flat_2d_array *Weights = (flat_2d_array *)malloc(sizeof(flat_2d_array));
    flat_2d_array *PreviousWeights = (flat_2d_array *)malloc(sizeof(flat_2d_array));
    
	u32 NeuronCount = 0;

    Data->RowCount = LayerCount;
    Delta->RowCount = LayerCount;
    
    u32 *RowPointer = (u32 *)malloc(LayerCount * sizeof(r32));
	for (u32 i = 0;
         i < LayerCount;
         i++)
	{
	    RowPointer[i] = NeuronCount;
		NeuronCount += LayerSizes[i];
	}

    Data->RowPointer = RowPointer;
    Delta->RowPointer = RowPointer;

    Data->MemberCount = NeuronCount;
    Delta->MemberCount = NeuronCount;

    Data->Array = (r32 *)malloc(NeuronCount * sizeof(r32));
    Delta->Array = (r32 *)malloc(NeuronCount * sizeof(r32));

    SeedArrayRandomly(Data->Array, NeuronCount);
    SeedArrayRandomly(Delta->Array, NeuronCount);

    u32 WeightCount = 0;

    RowPointer = (u32 *)malloc((LayerCount + 1) * sizeof(r32));
	RowPointer[0] = 0;
	for (u32 i=1;
         i < LayerCount;
         ++i)
	{
        RowPointer[i] = WeightCount;
		WeightCount += LayerSizes[i]*(LayerSizes[i-1]+1);
	}

    Weights->RowPointer = RowPointer;
    PreviousWeights->RowPointer = RowPointer;

    Weights->MemberCount = WeightCount;
    PreviousWeights->MemberCount = WeightCount;

    Weights->Array = (r32 *)malloc(WeightCount * sizeof(r32));
    PreviousWeights->Array = (r32 *)malloc(WeightCount * sizeof(r32));

    SeedArrayRandomly(Weights->Array, WeightCount);
    SeedArrayRandomly(PreviousWeights->Array, WeightCount);

    NeuralNetwork->Data = Data;
    NeuralNetwork->Delta = Delta;
    NeuralNetwork->Weights = Weights;
    NeuralNetwork->PreviousWeights = PreviousWeights;
}

internal r32
MeanSquareError(neural_network *NeuralNetwork, r32 *DataPoint)
{
    return 0;
}

internal void
FeedForward(neural_network *NeuralNetwork, r32 *DataPoint)
{
}

internal void
BackPropogate(neural_network *NeuralNetwork, r32 *DataPoint)
{
}

internal void
ShowResults(neural_network *NeuralNetwork, r32 *TestDataPoint, u32 TestIterationIndex)
{
}

internal void
FlatNN()
{
    srand(0);
    
    neural_network NeuralNetwork = {};

    u32 LayerSizes[] = LAYERSIZES;

    InitializeNeuralNetwork(&NeuralNetwork,
                            INPUTCOUNT, OUTPUTCOUNT, LayerSizes, ArrayCount(LayerSizes),
                            BETA, ALPHA, EPSILON, MAXITERATIONS);
    
    Assert(ArrayCount(TrainingData) % (NeuralNetwork.InputCount + NeuralNetwork.OutputCount) == 0);

    u32 TrainingDataPointCount = ArrayCount(TrainingData) / (NeuralNetwork.InputCount + NeuralNetwork.OutputCount);


    for(u32 IterationIndex = 0;
        IterationIndex < NeuralNetwork.MaximumIterations;
        ++IterationIndex)
    {
        r32 *TrainDataPoint = TrainingData +
            ((IterationIndex % TrainingDataPointCount) *
             (NeuralNetwork.InputCount + NeuralNetwork.OutputCount));

        BackPropogate(&NeuralNetwork, TrainDataPoint);
        r32 MSE = MeanSquareError(&NeuralNetwork, TrainDataPoint);
        
        //if(MSE < NeuralNetwork.Epsilon) break;
    }

    for(u32 TestIterationIndex = 0;
        TestIterationIndex < TrainingDataPointCount;
        ++TestIterationIndex)
    {
        r32 *TestDataPoint = TrainingData +
            ((TestIterationIndex % TrainingDataPointCount) *
             NeuralNetwork.InputCount + NeuralNetwork.OutputCount);
        FeedForward(&NeuralNetwork, TestDataPoint);
        ShowResults(&NeuralNetwork, TestDataPoint, TestIterationIndex);
    }
}
