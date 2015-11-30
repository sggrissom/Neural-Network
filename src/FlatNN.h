
#include <slib.h>

//NOTE(steven):Data[Layer][neuron]
//Weights[Layer][Neuron][WeightsForPrevLayerNeuron]

struct neural_network
{
    flat_2d_array *Data;
    flat_2d_array *Delta;
    flat_2d_array *Weights;
    
    u32 InputCount;
    u32 OutputCount;
    u32 *LayerSizes;
    u32 LayerCount;
    u32 MaximumIterations;
    r32 Beta;
    r32 Alpha;
    r32 Epsilon;
};

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>

internal r32
Sigmoid(r32 X)
{
    r32 Result = 1.0f/(1.0f + exp(-X));
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
        *(Array + ArrayIndex) = RandomNumber;
    }
}

struct flat_2d_array
{
    type *Array;
    u32 *Rowptr;
    u32 MemberCount;
    u32 RowCount;
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

    
}

internal r32
MeanSquareError(neural_network *NeuralNetwork, r32 *DataPoint)
{
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
FlatNN()
{
    srand(0);
    
    r32 TrainingData[] = {
        0,0,0,
        0,1,1,
        1,0,1,
        1,1,0,
    };
    
    neural_network NeuralNetwork = {};

    #define INPUTCOUNT 2
    #define OUTPUTCOUNT 1
#define LAYERSIZES {2,3,1}
    #define BETA 0.3f
    #define ALPHA 0.1f
    #define EPSILON 0.0001f
    #define MAXITERATIONS 500000

    InitializeNeuralNetwork(&NeuralNetwork,
                            INPUTCOUNT, OUTPUTCOUNT, LAYERSIZES, ArrayCount(LAYERSIZES),
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
