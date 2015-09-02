
#include "../sgg/slib.h"

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
        printf("float %d: %f\n", ArrayIndex, *Array);
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
    SeedArrayRandomly(NeuralNetwork->Data, DataArraySize);
    SeedArrayRandomly(NeuralNetwork->Weights, WeightArraySize);

    PrintArray(NeuralNetwork->Data, DataArraySize);
    PrintArray(NeuralNetwork->Weights, WeightArraySize);
}

internal r32 *
FeedForward(neural_network *NeuralNetwork, r32 *DataPoint)
{
    return 0;
}

internal void
BackPropogate(neural_network *NeuralNetwork, r32 *DataPoint)
{
}

s32 main()
{
    srand((u32)time(0));
    
    r32 TrainingData[] = {0,0,0,0,
                         0,0,1,1,
                         0,1,0,1,
                         1,0,0,1,
                         0,1,1,0,
                         1,0,1,0,
                         1,1,0,0,
                         1,1,1,1,};
    
    r32 TestData[] = {0,0,0,
                     0,0,1,
                     0,1,0,
                     1,0,0,
                     0,1,1,
                     1,0,1,
                     1,1,0,
                     1,1,1,};
    
    neural_network xor = {};
    xor.InputCount = 3;
    xor.OutputCount = 1;
    
    u32 LayerSizes[] = {3,3,2,1};
    xor.LayerSizes = LayerSizes;
    xor.LayerCount = ArrayCount(LayerSizes);

    xor.Beta = 0.3f;
    xor.Alpha = 0.1f;
    xor.Epsilon = 0.0001f;

    xor.MaximumIterations = 2000000;

    Assert(ArrayCount(TrainingData) % (xor.InputCount + xor.OutputCount) == 0);
    Assert(ArrayCount(TestData) % (xor.InputCount) == 0);

    u32 TrainingDataPointCount = ArrayCount(TrainingData) / (xor.InputCount + xor.OutputCount);
    u32 TestDataPointCount = ArrayCount(TestData) / (xor.InputCount);

    InitializeNeuralNetwork(&xor);

    for(u32 IterationIndex = 0;
        IterationIndex < xor.MaximumIterations;
        ++IterationIndex)
    {
        r32 *TrainDataPoint = TrainingData + IterationIndex % TrainingDataPointCount;
        FeedForward(&xor, TrainDataPoint);
        
    }

    for(u32 TestIterationIndex = 0;
        TestIterationIndex < TestDataPointCount;
        ++TestIterationIndex)
    {
        r32 *TestDataPoint = TestData + TestIterationIndex % TestDataPointCount;
        r32 *NeuralNetworkOutput = FeedForward(&xor, TestDataPoint);
        for(u32 OutputIndex = 0;
            OutputIndex < xor.OutputCount;
            ++OutputIndex)
        {
            //r32 OutputValue = *(NeuralNetworkOutput + OutputIndex);
        }
    }
    
    return 1;
}
