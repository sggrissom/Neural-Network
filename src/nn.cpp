
#include <time.h>
#include <math.h>

#include <slib.h>

#define LAYERSIZES {2,3,1}
#define BETA 0.3f
#define ALPHA 0.1f
#define EPSILON 0.0001f
#define MAX_ITERATIONS 500000
#define TEST_ITERATIONS 1000

global r32 TrainingData[] = {
    0,0,0,
    0,1,1,
    1,0,1,
    1,1,0,
};
    
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

internal void
FeedForward(neural_network *NeuralNetwork, r32 *DataPoint)
{
    r32 *Data = NeuralNetwork->Data;
    r32 *Weights = NeuralNetwork->Weights;
    u32 *DataRowPtr = NeuralNetwork->DataRowPointer;
    u32 *WeightsRowPtr = NeuralNetwork->WeightsRowPointer;
    u32 *LayerSizes = NeuralNetwork->LayerSizes;
    
	for (u32 InputIndex = 0;
         InputIndex < NeuralNetwork->InputCount;
         ++InputIndex)
	{
		Data[DataRowPtr[0]+InputIndex] = DataPoint[InputIndex];
	}

	for(u32 LayerIndex = 1;
        LayerIndex < NeuralNetwork->LayerCount;
        ++LayerIndex)
	{
		for(u32 NeuronIndex = 0;
            NeuronIndex < LayerSizes[LayerIndex];
            ++NeuronIndex)
		{
			r32 Sum = 0.0f;
			for(u32 PrevNeuronIndex = 0;
                PrevNeuronIndex < LayerSizes[LayerIndex-1];
                ++PrevNeuronIndex)
			{
                r32 NeuronValue = Data[DataRowPtr[LayerIndex-1]+PrevNeuronIndex];
                r32 WeightValue = Weights[WeightsRowPtr[LayerIndex] +
                                          (LayerSizes[LayerIndex-1]+1)*NeuronIndex +
                                          PrevNeuronIndex];
				Sum += NeuronValue * WeightValue;
			}
            
			Sum += Weights[WeightsRowPtr[LayerIndex] +
                                          (LayerSizes[LayerIndex-1]+1)*NeuronIndex +
                                          LayerSizes[LayerIndex-1]];
            
			Data[DataRowPtr[LayerIndex]+NeuronIndex] = (1.0f/(1.0f+(r32)exp(-Sum)));
		}
	}
}

internal void
BackPropogate(neural_network *NeuralNetwork, r32 *DataPoint, r32 *Target)
{
    FeedForward(NeuralNetwork, DataPoint);

    r32 *Data = NeuralNetwork->Data;
    r32 *Delta = NeuralNetwork->Delta;
    r32 *Weights = NeuralNetwork->Weights;
    r32 *WeightDelta = NeuralNetwork->WeightDelta;
    u32 *DataRowPtr = NeuralNetwork->DataRowPointer;
    u32 *WeightsRowPtr = NeuralNetwork->WeightsRowPointer;
    u32 *LayerSizes = NeuralNetwork->LayerSizes;
    u32 LayerCount = NeuralNetwork->LayerCount;
    r32 Alpha = NeuralNetwork->Alpha;
    r32 Beta = NeuralNetwork->Beta;

	for (u32 LayerIndex = 0;
         LayerIndex < LayerSizes[(LayerCount)-1];
         ++LayerIndex)
	{
        r32 Temp = Data[DataRowPtr[(LayerCount)-1]+LayerIndex];
		Delta[DataRowPtr[(LayerCount)-1]+LayerIndex] =
            Temp * (1 - Temp) *
            (Target[LayerIndex] - Data[DataRowPtr[(LayerCount)-1]+LayerIndex]);
	}

	for(u32 LayerIndex = LayerCount-2;
        LayerIndex > 0;
        --LayerIndex)
	{
		for(u32 NeuronIndex = 0;
            NeuronIndex < LayerSizes[LayerIndex];
            ++NeuronIndex)
		{
			r32 Sum = 0.0f;
			for(u32 NextNeuronIndex=0;
                NextNeuronIndex < LayerSizes[LayerIndex+1];
                ++NextNeuronIndex)
			{
                r32 DeltaValue = Delta[DataRowPtr[LayerIndex+1]+NextNeuronIndex];
                r32 WeightValue = Weights[WeightsRowPtr[LayerIndex + 1] +
                                          (LayerSizes[LayerIndex]+1)*NeuronIndex +
                                          NextNeuronIndex];

                Sum += DeltaValue * WeightValue;
			}

            Delta[DataRowPtr[LayerIndex] + NeuronIndex] =
                Data[DataRowPtr[LayerIndex] + NeuronIndex] *
                (1 - Data[DataRowPtr[LayerIndex] + NeuronIndex]) *
                Sum;
		}
	}

	for(u32 LayerIndex = 1;
        LayerIndex < LayerCount;
        ++LayerIndex)
	{
        for(u32 NeuronIndex = 0;
            NeuronIndex < LayerSizes[LayerIndex];
            ++NeuronIndex)
		{
            u32 PrevLayerSize = LayerSizes[LayerIndex-1]+1;
            for(u32 PrevNeuronIndex = 0;
                PrevNeuronIndex < LayerSizes[LayerIndex-1];
                ++PrevNeuronIndex)
			{
                Weights[WeightsRowPtr[LayerIndex] + PrevLayerSize * NeuronIndex + PrevNeuronIndex] +=
                    (Alpha) * WeightDelta[WeightsRowPtr[LayerIndex] +
                                          PrevLayerSize * NeuronIndex + PrevNeuronIndex];
			}

            Weights[WeightsRowPtr[LayerIndex] + PrevLayerSize * NeuronIndex + LayerSizes[LayerIndex-1]] +=
                Alpha * WeightDelta[WeightsRowPtr[LayerIndex] +
                                    PrevLayerSize * NeuronIndex + LayerSizes[LayerIndex-1]];
		}
	}

	for(u32 LayerIndex = 1;
        LayerIndex < LayerCount;
        ++LayerIndex)
	{
        for(u32 NeuronIndex = 0;
            NeuronIndex < LayerSizes[LayerIndex];
            ++NeuronIndex)
		{
            u32 PrevLayerSize = LayerSizes[LayerIndex-1]+1;
            for(u32 PrevNeuronIndex = 0;
                PrevNeuronIndex < LayerSizes[LayerIndex-1];
                ++PrevNeuronIndex)
			{
                WeightDelta[WeightsRowPtr[LayerIndex] + PrevLayerSize * NeuronIndex + PrevNeuronIndex] +=
                    Beta * Delta[DataRowPtr[LayerIndex]+NeuronIndex] *
                    Data[DataRowPtr[LayerIndex-1] + PrevNeuronIndex];
                Weights[WeightsRowPtr[LayerIndex] + PrevLayerSize * NeuronIndex + PrevNeuronIndex] +=
                    WeightDelta[WeightsRowPtr[LayerIndex] +
                                PrevLayerSize * NeuronIndex + PrevNeuronIndex];
			}
            
            WeightDelta[WeightsRowPtr[LayerIndex] + PrevLayerSize * NeuronIndex + LayerSizes[LayerIndex-1]] +=
                Beta * Delta[DataRowPtr[LayerIndex]+NeuronIndex] *
                Data[DataRowPtr[LayerIndex-1] + LayerSizes[LayerIndex-1]];
            Weights[WeightsRowPtr[LayerIndex] + PrevLayerSize * NeuronIndex + LayerSizes[LayerIndex-1]] +=
                WeightDelta[WeightsRowPtr[LayerIndex] +
                            PrevLayerSize * NeuronIndex + LayerSizes[LayerIndex-1]];
		}
	}
}

s32 main()
{
    srand(0);
    
    neural_network NeuralNetwork = {};
    
    u32 LayerSizes[] = LAYERSIZES;
    NeuralNetwork.LayerSizes = LayerSizes;
    NeuralNetwork.LayerCount = ArrayCount(LayerSizes);
    NeuralNetwork.InputCount = LayerSizes[0];
    NeuralNetwork.OutputCount = LayerSizes[NeuralNetwork.LayerCount - 1];
    NeuralNetwork.Beta = BETA;
    NeuralNetwork.Alpha = ALPHA;
    NeuralNetwork.Epsilon = EPSILON;

    u32 MaximumIterations =  MAX_ITERATIONS;

    u32 InputCount = NeuralNetwork.InputCount;
    u32 OutputCount = NeuralNetwork.OutputCount;
    u32 LayerCount = NeuralNetwork.LayerCount;

    Assert(ArrayCount(TrainingData) % (InputCount + OutputCount) == 0);

    u32 TrainingDataPointCount = ArrayCount(TrainingData) / (InputCount + OutputCount);
    u32 TrainDataPitch = InputCount + OutputCount;

    InitializeNetwork(&NeuralNetwork);

    clock_t startTime = clock();
    
    for(u32 IterationIndex = 0;
        IterationIndex < MaximumIterations;
        ++IterationIndex)
    {
        r32 *DataPoint = &TrainingData[(IterationIndex%TrainingDataPointCount)*TrainDataPitch];
        r32 *Target = DataPoint + InputCount;
        BackPropogate(&NeuralNetwork, DataPoint, Target);
    }

    clock_t endTime = clock();
    clock_t clockTicksTaken = endTime - startTime;
    r32 timeInSeconds = clockTicksTaken / (r32) CLOCKS_PER_SEC;
    
    printf("%f s\n", timeInSeconds);

#if 0
    for(u32 i=0;
        i < TrainingDataPointCount;
        ++i)
	{
		FeedForward(&NeuralNetwork,
                    &TrainingData[i*TrainDataPitch]);
		for (u32 j=0; j < InputCount; j++)
		{
			r32 Value = TrainingData[i*(TrainDataPitch)+j];
            printf("%f ", Value);
		}

        printf("\n");
        printf("Answer: %f\n", TrainingData[i * TrainDataPitch + InputCount]);
        printf("Guess: %f\n", NeuralNetwork.Data[NeuralNetwork.DataRowPointer[LayerCount-1]]);
	}
    
    printf("\n\n\n");
#endif

    u32 Correct = 0;
    u32 Incorrect = 0;

	for(u32 i=0; i<TEST_ITERATIONS; i++)
	{
		FeedForward(&NeuralNetwork,
                    &TrainingData[((i%TrainingDataPointCount)*TrainDataPitch)]);
		u32 Actual = (u32)(TrainingData[(i%TrainingDataPointCount)*TrainDataPitch + InputCount]);
		r32 Guess = NeuralNetwork.Data[NeuralNetwork.DataRowPointer[LayerCount-1]];
		u32 Prediction = (u32)(Guess + 0.5f);
		if (Prediction == Actual)
		{
			++Correct;
		} else {
		   ++Incorrect;
		}
	}

    r32 Accuracy = ((r32)Correct)/((r32)(Correct+Incorrect));

    printf("Correct: %d\n", Correct);
    printf("Incorrect: %d\n", Incorrect);
    printf("Accuracy: %.2f%%\n", Accuracy*100);
    
    return 1;
}
