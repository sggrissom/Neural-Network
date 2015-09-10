
#include "../slib/slib.h"

//NOTE(steven):Data[Layer][neuron]
//Weights[Layer][Neuron][WeightsForPrevLayerNeuron]

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
#include <string.h>

internal r32
Sigmoid(r32 X)
{
    r32 Result = 1.0f/(1.0f + (r32)pow(e32,-X));

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
        r32 RandomNumber = (r32)rand() / (RAND_MAX/2) - 1;
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
    memset(NeuralNetwork->Data, 0, DataArraySize * sizeof(r32));
    
    NeuralNetwork->Weights = (r32*)malloc(sizeof(r32) * WeightArraySize);
    SeedArrayRandomly(NeuralNetwork->Weights, WeightArraySize);

    PrintArray(NeuralNetwork->Data, DataArraySize);
    PrintArray(NeuralNetwork->Weights, WeightArraySize);
}

internal r32 *
GetNeuralNetworkOutput(neural_network *NeuralNetwork)
{
    u32 OutputLocation = 0;

    for(u32 LayerIndex = 0;
        LayerIndex < NeuralNetwork->LayerCount;
        ++LayerIndex)
    {
        OutputLocation += *(NeuralNetwork->LayerSizes + LayerIndex);
    }

    return NeuralNetwork->Data + OutputLocation;
}

internal r32
MeanSquareError(neural_network *NeuralNetwork, r32 *DataPoint)
{
    r32 MSE = 0;

    r32 *Output = GetNeuralNetworkOutput(NeuralNetwork);

    for(u32 OutputIndex = 0;
        OutputIndex < NeuralNetwork->OutputCount;
        ++OutputIndex)
    {
        r32 TargetValue = *(DataPoint + OutputIndex);
        r32 OutputValue = *(Output + OutputIndex);
        r32 Delta = TargetValue - OutputValue;
        MSE += Delta * Delta;
    }

    return MSE * 0.5f;
}

internal void
FeedForward(neural_network *NeuralNetwork, r32 *DataPoint)
{
    u32 const LayerCount = NeuralNetwork->LayerCount;
    u32 LayerSizes[LayerCount];

    for(u32 LayerIndex = 0;
        LayerIndex < LayerCount;
        ++LayerIndex)
    {
        LayerSizes[LayerIndex] = *(NeuralNetwork->LayerSizes + LayerIndex);
    }

    r32 *Data = NeuralNetwork->Data;
    r32 *Input = DataPoint;
    
    for(u32 InputIndex = 0;
        InputIndex < LayerSizes[0];
        ++InputIndex)
    {
        *Data++ = *Input++;
    }

    r32 *WeightPointer = NeuralNetwork->Weights;
    r32 *PrevNeuronPointer = NeuralNetwork->Data;

    r32 *LayerPointer = NeuralNetwork->Data + LayerSizes[0];

    u32 LayerSize = 0, PrevLayerSize = 0;

    //NOTE(steven): for each neuron, it's value is equal to the sum of the previous layer's
    //neuron values multiplied by their weight values, summed together (including a bias neuron).
    //This value then has the sigmoid function applied to it.
    
    for(u32 LayerIndex = 1;
        LayerIndex < NeuralNetwork->LayerCount;
        ++LayerIndex)
    {
        LayerSize = LayerSizes[LayerIndex];
        PrevLayerSize = LayerSizes[LayerIndex - 1];

        r32 *NeuronPointer = LayerPointer;
        for(u32 NeuronIndex = 0;
            NeuronIndex < LayerSize;
            ++NeuronIndex)
        {
            r32 Sum = 0;

            for(u32 PrevLayerNeuronIndex = 0;
                PrevLayerNeuronIndex < PrevLayerSize;
                ++PrevLayerNeuronIndex)
            {
                r32 PrevNeuronValue = *PrevNeuronPointer++;
                r32 NeuronWeight = *WeightPointer++;
                Sum += PrevNeuronValue * NeuronWeight;
            }
            
            Sum += *WeightPointer++;
            *NeuronPointer++ = Sigmoid(Sum);
        }
        
        LayerPointer += LayerSize;
        PrevNeuronPointer = LayerPointer;
    }    
}

internal void
BackPropogate(neural_network *NeuralNetwork, r32 *DataPoint)
{
    FeedForward(NeuralNetwork, DataPoint);

    //NOTE(steven): find delta
    
    //NOTE(steven): apply momentum - does nothing if (Alpha == 0)
    
    //NOTE(steven): adjust weights
    
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

    NeuralNetwork.MaximumIterations = 20;

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
        PrintArray(NeuralNetwork.Data, 9);
        r32 MSE = MeanSquareError(&NeuralNetwork, TrainDataPoint);
        printf("mse: %f\n", MSE);
    }

    for(u32 TestIterationIndex = 0;
        TestIterationIndex < TestDataPointCount;
        ++TestIterationIndex)
    {
        r32 *TestDataPoint = TestData + TestIterationIndex % TestDataPointCount;
        FeedForward(&NeuralNetwork, TestDataPoint);
        for(u32 OutputIndex = 0;
            OutputIndex < NeuralNetwork.OutputCount;
            ++OutputIndex)
        {
            //r32 OutputValue = *(NeuralNetworkOutput + OutputIndex);
        }
    }
    
    return 1;
}
