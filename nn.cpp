
#include "../slib/slib.h"

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
    r32 *Data;
    r32 *Weights;
};

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

internal r32
Sigmoid(r32 X)
{
    return 1.0f/(1.0f + (r32)pow(e32,-X));
}

internal void
SeedArrayRandomly(r32 *Array, u32 ArraySize)
{
    for(u32 ArrayIndex = 0;
        ArrayIndex < ArraySize;
        ++ArrayIndex)
    {
        r32 RandomNumber = (r32)rand();
        *(Array + ArrayIndex) = RandomNumber;
    }
}

internal void
PrintArray(r32 *Array, u32 ArraySize)
{
    printf("\n\nPrint Array\n\n");
    for(u32 ArrayIndex = 0;
        ArrayIndex < ArraySize;
        ++ArrayIndex)
    {
        printf("float %d: %f\n", ArrayIndex, *(Array + ArrayIndex));
    }
}

internal void
InitializeNeuralNetwork(neural_network *NeuralNetwork)
{
    
    u32 DataArraySize = 0, WeightArraySize = 0, LayerSize = 0, PrevLayerSize = 0;
    for(u32 LayerIndex = 0;
        LayerIndex < NeuralNetwork->LayerCount;
        ++LayerIndex)
    {
        LayerSize = *(NeuralNetwork->LayerSizes + LayerIndex);
        DataArraySize += LayerSize;
        WeightArraySize += LayerSize * PrevLayerSize + PrevLayerSize;
        PrevLayerSize = LayerSize;
    }

    NeuralNetwork->Data = (r32*)malloc(sizeof(r32) * DataArraySize);
    NeuralNetwork->Weights = (r32*)malloc(sizeof(r32) * WeightArraySize);
    
    SeedArrayRandomly(NeuralNetwork->Weights, WeightArraySize);
}

#define NeuralNetworkData(nn, i, j) (*(i*nn->LayerCount + j))
#define NeuralNetworkWeight(nn, i, j, k) (*(i*nn->LayerCount*nn-> + j*(*nn->LayerSizes) + k))

internal void
FeedForward(neural_network *NeuralNetwork, r32 *DataPoint)
{
    *InputPoint = NeuralNetwork->Data;
    for(u32 InputIndex = 0;
        InputIndex < NeuralNetwork->InputCount;
        ++InputIndex)
    {
        *InputPoint++ = *DataPoint++
    }

    for(u32 LayerIndex = 1;
        LayerIndex < NeuralNetwork->LayerCount;
        ++LayerIndex)
    {
        for(u32 NeuronIndex = 0;
            NeuronIndex < *(NeuralNetwork->LayerSizes + LayerIndex);
            ++NeuronIndex)
        {
            r32 sum = 0;
            for(u32 PrevLayerNeuronIndex = 0;
                PrevLayerNeuronIndex < *(NeuralNetwork->LayerSizes + LayerIndex - 1);
                ++PrevLayerNeuronIndex)
            {
                sum += 
            }
            sum += *(NeuralNetwork->Weights);
            *NeuralNetwork->Data = Sigmoid(sum);
        }
    }
}

internal void
BackPropogate(neural_network *NeuralNetwork, r32 *DataPoint)
{
}

s32 main()
{
    srand((u32)time(0));
    
    r32 TrainingData[] = {
        0,0,0,0,
        0,0,1,1,
        0,1,0,1,
        1,0,0,1,
        0,1,1,0,
        1,0,1,0,
        1,1,0,0,
        1,1,1,1,
    };
    
    r32 TestData[] = {
        0,0,0,
        0,0,1,
        0,1,0,
        1,0,0,
        0,1,1,
        1,0,1,
        1,1,0,
        1,1,1,
    };
    
    neural_network NeuralNetwork = {};
    NeuralNetwork.InputCount = 3;
    NeuralNetwork.OutputCount = 1;
    
    u32 LayerSizes[] = {3,3,2,1};
    NeuralNetwork.LayerSizes = LayerSizes;
    NeuralNetwork.LayerCount = ArrayCount(LayerSizes);

    NeuralNetwork.Beta = 0.3f;
    NeuralNetwork.Alpha = 0.1f;
    NeuralNetwork.Epsilon = 0.0001f;

    NeuralNetwork.MaximumIterations = 2000000;

    Assert(ArrayCount(TrainingData) % (NeuralNetwork.InputCount + NeuralNetwork.OutputCount) == 0);
    Assert(ArrayCount(TestData) % (NeuralNetwork.InputCount) == 0);

    u32 TrainingDataPointCount = ArrayCount(TrainingData) / (NeuralNetwork.InputCount + NeuralNetwork.OutputCount);
    u32 TestDataPointCount = ArrayCount(TestData) / (NeuralNetwork.InputCount);

    InitializeNeuralNetwork(&NeuralNetwork);

    for(u32 IterationIndex = 0;
        IterationIndex < NeuralNetwork.MaximumIterations;
        ++IterationIndex)
    {
        r32 *TrainDataPoint = TrainingData + IterationIndex % TrainingDataPointCount;
        FeedForward(&NeuralNetwork, TrainDataPoint);
        
    }

    for(u32 TestIterationIndex = 0;
        TestIterationIndex < TestDataPointCount;
        ++TestIterationIndex)
    {
        r32 *TestDataPoint = TestData + TestIterationIndex % TestDataPointCount;
        r32 *NeuralNetworkOutput = FeedForward(&NeuralNetwork, TestDataPoint);
        for(u32 OutputIndex = 0;
            OutputIndex < NeuralNetwork.OutputCount;
            ++OutputIndex)
        {
            //r32 OutputValue = *(NeuralNetworkOutput + OutputIndex);
        }
    }
    
    return 1;
}
