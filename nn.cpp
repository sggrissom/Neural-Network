
#include "../sgg/sgg.h"

struct NeuralNetwork
{
    u32 InputCount;
    u32 OutputCount;
    u32 *LayerSizes;
    u32 MaximumIterations;
    r32 Beta;
    r32 Alpha;
    r32 Epsilon;
    r32 *Data;
};

internal r32
Sigmoid(r32 X)
{
    return 0;
}

internal r32 *
FeedForward(NeuralNetwork *nn, r32 *DataPoint)
{
    return 0;
}

internal void
BackPropogate(NeuralNetwork *nn, r32 *DataPoint)
{
}

s32 main()
{
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
    
    NeuralNetwork xor = {};
    xor.InputCount = 3;
    xor.OutputCount = 1;
    
    u32 LayerSizes[] = {3,3,2,1};
    xor.LayerSizes = LayerSizes;

    xor.Beta = 0.3f;
    xor.Alpha = 0.1f;
    xor.Epsilon = 0.0001f;

    xor.MaximumIterations = 2000000;

    Assert(ArrayCount(TrainingData) % (xor.InputCount + xor.OutputCount) == 0);
    Assert(ArrayCount(TestData) % (xor.InputCount) == 0);

    u32 TrainingDataPointCount = ArrayCount(TrainingData) / (xor.InputCount + xor.OutputCount);
    u32 TestDataPointCount = ArrayCount(TestData) / (xor.InputCount);
    

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
            r32 OutputValue = *(NeuralNetworkOutput + OutputIndex);
        }
    }
    
    return 1;
}
