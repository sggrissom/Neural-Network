/* ========================================================================
   File: FlatNN.cpp
   Date: 2015-12-15
   Creator: Steven Grissom
   ======================================================================== */


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
            u32 PrevLayerSize = LayerSizes[LayerIndex-1]+1;
            while(PrevLayerSize&15)
            {
                ++PrevLayerSize;
            }
			for(u32 PrevNeuronIndex = 0;
                PrevNeuronIndex < LayerSizes[LayerIndex-1];
                ++PrevNeuronIndex)
			{
                r32 NeuronValue = Data[DataRowPtr[LayerIndex-1]+PrevNeuronIndex];
                r32 WeightValue = Weights[WeightsRowPtr[LayerIndex] +
                                          (PrevLayerSize)*NeuronIndex +
                                          PrevNeuronIndex];
				Sum += NeuronValue * WeightValue;
			}
            
			Sum += Weights[WeightsRowPtr[LayerIndex] +
                                          (PrevLayerSize)*NeuronIndex +
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
        u32 CurrLayerSize = LayerSizes[LayerIndex]+1;
        while(CurrLayerSize&15)
        {
            ++CurrLayerSize;
        }
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
                                          CurrLayerSize*NextNeuronIndex +
                                          NeuronIndex];

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
            while(PrevLayerSize&15)
            {
                ++PrevLayerSize;
            }
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
            while(PrevLayerSize&15)
            {
                ++PrevLayerSize;
            }
            for(u32 PrevNeuronIndex = 0;
                PrevNeuronIndex < LayerSizes[LayerIndex-1];
                ++PrevNeuronIndex)
			{
                WeightDelta[WeightsRowPtr[LayerIndex] +
                            (PrevLayerSize * NeuronIndex)
                            + PrevNeuronIndex] =
                    Beta * Delta[DataRowPtr[LayerIndex]+NeuronIndex] *
                    Data[DataRowPtr[LayerIndex-1] + PrevNeuronIndex];
                
                Weights[WeightsRowPtr[LayerIndex] +
                        (PrevLayerSize * NeuronIndex) +
                        PrevNeuronIndex] +=
                    WeightDelta[WeightsRowPtr[LayerIndex] +
                                PrevLayerSize * NeuronIndex + PrevNeuronIndex];
			}
            
            WeightDelta[WeightsRowPtr[LayerIndex] +
                        (PrevLayerSize * NeuronIndex)
                        + LayerSizes[LayerIndex-1]] =
                Beta * Delta[DataRowPtr[LayerIndex]+NeuronIndex];
            
            Weights[WeightsRowPtr[LayerIndex] +
                    (PrevLayerSize * NeuronIndex) +
                    LayerSizes[LayerIndex-1]] +=
                WeightDelta[WeightsRowPtr[LayerIndex] +
                            PrevLayerSize * NeuronIndex + LayerSizes[LayerIndex-1]];
		}
	}    
}
