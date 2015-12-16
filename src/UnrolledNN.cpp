/* ========================================================================
   File: UnrolledNN.cpp
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

    {
        u32 InputIndex = 0;
        for (;
             InputIndex+3 < NeuralNetwork->InputCount;
             InputIndex += 4)
        {
            Data[DataRowPtr[0]+InputIndex] = DataPoint[InputIndex];
            Data[DataRowPtr[0]+InputIndex+1] = DataPoint[InputIndex+1];
            Data[DataRowPtr[0]+InputIndex+2] = DataPoint[InputIndex+2];
            Data[DataRowPtr[0]+InputIndex+3] = DataPoint[InputIndex+3];
        }
    
        for (;
             InputIndex < NeuralNetwork->InputCount;
             ++InputIndex)
        {
            Data[DataRowPtr[0]+InputIndex] = DataPoint[InputIndex];
        }
    }

    for(u32 LayerIndex = 1;
        LayerIndex < NeuralNetwork->LayerCount;
        ++LayerIndex)
	{
        u32 PrevLayerSize = LayerSizes[LayerIndex-1]+1;
        while(PrevLayerSize&15)
        {
            ++PrevLayerSize;
        }
            
        {
            u32 NeuronIndex = 0;
            for(;
                NeuronIndex+3 < LayerSizes[LayerIndex];
                NeuronIndex+=4)
            {
                r32 c00 = 0.0f;
                r32 c01 = 0.0f;
                r32 c02 = 0.0f;
                r32 c03 = 0.0f;
             
                r32 c10 = 0.0f;
                r32 c11 = 0.0f;
                r32 c12 = 0.0f;
                r32 c13 = 0.0f;
             
                r32 c20 = 0.0f;
                r32 c21 = 0.0f;
                r32 c22 = 0.0f;
                r32 c23 = 0.0f;
             
                r32 c30 = 0.0f;
                r32 c31 = 0.0f;
                r32 c32 = 0.0f;
                r32 c33 = 0.0f;
            
                {
                    u32 PrevNeuronIndex = 0;
                    for(;
                        PrevNeuronIndex+3 < LayerSizes[LayerIndex-1];
                        PrevNeuronIndex+=4)
                    {
                        r32 Value0 = (Data[DataRowPtr[LayerIndex-1]+PrevNeuronIndex]);
                        r32 Value1 = (Data[DataRowPtr[LayerIndex-1]+PrevNeuronIndex+1]);
                        r32 Value2 = (Data[DataRowPtr[LayerIndex-1]+PrevNeuronIndex+2]);
                        r32 Value3 = (Data[DataRowPtr[LayerIndex-1]+PrevNeuronIndex+3]);

                        c00 +=  Value0 *
                            (Weights[WeightsRowPtr[LayerIndex] +
                                     (PrevLayerSize)*(NeuronIndex+0) +
                                     PrevNeuronIndex]);
                        c01 +=  Value1 *
                            (Weights[WeightsRowPtr[LayerIndex] +
                                     (PrevLayerSize)*(NeuronIndex+0) +
                                     PrevNeuronIndex+1]);
                        c02 += Value2 *
                            (Weights[WeightsRowPtr[LayerIndex] +
                                     (PrevLayerSize)*(NeuronIndex+0) +
                                     PrevNeuronIndex+2]);
                        c03 += Value3 *
                            (Weights[WeightsRowPtr[LayerIndex] +
                                     (PrevLayerSize)*(NeuronIndex+0) +
                                     PrevNeuronIndex+3]);

                        c10 += Value0 *
                            (Weights[WeightsRowPtr[LayerIndex] +
                                     (PrevLayerSize)*(NeuronIndex+1) +
                                     PrevNeuronIndex]);
                        c11 += Value1 *
                            (Weights[WeightsRowPtr[LayerIndex] +
                                     (PrevLayerSize)*(NeuronIndex+1) +
                                     PrevNeuronIndex+1]);
                        c12 += Value2 *
                            (Weights[WeightsRowPtr[LayerIndex] +
                                     (PrevLayerSize)*(NeuronIndex+1) +
                                     PrevNeuronIndex+2]);
                        c13 += Value3 *
                            (Weights[WeightsRowPtr[LayerIndex] +
                                     (PrevLayerSize)*(NeuronIndex+1) +
                                     PrevNeuronIndex+3]);
                        
                        c20 += Value0 *
                            (Weights[WeightsRowPtr[LayerIndex] +
                                     (PrevLayerSize)*(NeuronIndex+2) +
                                     PrevNeuronIndex]);
                        c21 += Value1 *
                            (Weights[WeightsRowPtr[LayerIndex] +
                                     (PrevLayerSize)*(NeuronIndex+2) +
                                     PrevNeuronIndex+1]);
                        c22 += Value2 *
                            (Weights[WeightsRowPtr[LayerIndex] +
                                     (PrevLayerSize)*(NeuronIndex+2) +
                                     PrevNeuronIndex+2]);
                        c23 += Value3 *
                            (Weights[WeightsRowPtr[LayerIndex] +
                                     (PrevLayerSize)*(NeuronIndex+2) +
                                     PrevNeuronIndex+3]);
                        
                        c30 += Value0 *
                            (Weights[WeightsRowPtr[LayerIndex] +
                                     (PrevLayerSize)*(NeuronIndex+3) +
                                     PrevNeuronIndex]);
                        c31 += Value1 *
                            (Weights[WeightsRowPtr[LayerIndex] +
                                     (PrevLayerSize)*(NeuronIndex+3) +
                                     PrevNeuronIndex+1]);
                        c32 += Value2 *
                            (Weights[WeightsRowPtr[LayerIndex] +
                                     (PrevLayerSize)*(NeuronIndex+3) +
                                     PrevNeuronIndex+2]);
                        c33 += Value3 *
                            (Weights[WeightsRowPtr[LayerIndex] +
                                     (PrevLayerSize)*(NeuronIndex+3) +
                                     PrevNeuronIndex+3]);
                    }

                    for(;
                        PrevNeuronIndex < LayerSizes[LayerIndex-1];
                        ++PrevNeuronIndex)
                    {
                        InvalidCodePath;
                        
                        c00 += (Data[DataRowPtr[LayerIndex-1]+PrevNeuronIndex]) *
                            (Weights[WeightsRowPtr[LayerIndex] +
                                     (PrevLayerSize)*(NeuronIndex+0) +
                                     PrevNeuronIndex]);
                        c10 += (Data[DataRowPtr[LayerIndex-1]+PrevNeuronIndex]) *
                            (Weights[WeightsRowPtr[LayerIndex] +
                                     (PrevLayerSize)*(NeuronIndex+1) +
                                     PrevNeuronIndex]);
                        c20 += (Data[DataRowPtr[LayerIndex-1]+PrevNeuronIndex]) *
                            (Weights[WeightsRowPtr[LayerIndex] +
                                     (PrevLayerSize)*(NeuronIndex+2) +
                                     PrevNeuronIndex]);
                        c30 += (Data[DataRowPtr[LayerIndex-1]+PrevNeuronIndex]) *
                            (Weights[WeightsRowPtr[LayerIndex] +
                                     (PrevLayerSize)*(NeuronIndex+3) +
                                     PrevNeuronIndex]);
                    }
                }
                
                r32 Sum0 = c00 + c01 + c02 + c03 +
                    Weights[WeightsRowPtr[LayerIndex] +
                            (PrevLayerSize)*(NeuronIndex+0) +
                            LayerSizes[LayerIndex-1]];

                Data[DataRowPtr[LayerIndex]+(NeuronIndex+0)] = (1.0f/(1.0f+(r32)exp(-Sum0)));
                
                r32 Sum1 = c10 + c11 + c12 + c13 +
                    Weights[WeightsRowPtr[LayerIndex] +
                            (PrevLayerSize)*(NeuronIndex+1) +
                            LayerSizes[LayerIndex-1]];

                Data[DataRowPtr[LayerIndex]+(NeuronIndex+1)] = (1.0f/(1.0f+(r32)exp(-Sum1)));
                
                r32 Sum2 = c20 + c21 + c22 + c23 +
                    Weights[WeightsRowPtr[LayerIndex] +
                            (PrevLayerSize)*(NeuronIndex+2) +
                            LayerSizes[LayerIndex-1]];

                Data[DataRowPtr[LayerIndex]+(NeuronIndex+2)] = (1.0f/(1.0f+(r32)exp(-Sum2)));
                
                r32 Sum3 = c30 + c31 + c32 + c33 +
                    Weights[WeightsRowPtr[LayerIndex] +
                            (PrevLayerSize)*(NeuronIndex+3) +
                            LayerSizes[LayerIndex-1]];

                Data[DataRowPtr[LayerIndex]+(NeuronIndex+3)] = (1.0f/(1.0f+(r32)exp(-Sum3)));
            }
            
            for(;
                NeuronIndex < LayerSizes[LayerIndex];
                ++NeuronIndex)
            {
                r32 Sum = 0.0f;
                r32 c0 = 0.0f;
                r32 c1 = 0.0f;
                r32 c2 = 0.0f;
                r32 c3 = 0.0f;
            
                {
                    u32 PrevNeuronIndex = 0;
                    for(;
                        PrevNeuronIndex+3 < LayerSizes[LayerIndex-1];
                        PrevNeuronIndex+=4)
                    {
                        c0 += (Data[DataRowPtr[LayerIndex-1]+PrevNeuronIndex]) *
                            (Weights[WeightsRowPtr[LayerIndex] +
                                     (PrevLayerSize)*NeuronIndex +
                                     PrevNeuronIndex]);
                        c1 += (Data[DataRowPtr[LayerIndex-1]+PrevNeuronIndex+1]) *
                            (Weights[WeightsRowPtr[LayerIndex] +
                                     (PrevLayerSize)*NeuronIndex +
                                     PrevNeuronIndex+1]);
                        c2 += (Data[DataRowPtr[LayerIndex-1]+PrevNeuronIndex+2]) *
                            (Weights[WeightsRowPtr[LayerIndex] +
                                     (PrevLayerSize)*NeuronIndex +
                                     PrevNeuronIndex+2]);
                        c3 += (Data[DataRowPtr[LayerIndex-1]+PrevNeuronIndex+3]) *
                            (Weights[WeightsRowPtr[LayerIndex] +
                                     (PrevLayerSize)*NeuronIndex +
                                     PrevNeuronIndex+3]);
                    }
            
                    for(;
                        PrevNeuronIndex < LayerSizes[LayerIndex-1];
                        ++PrevNeuronIndex)
                    {
                        c0 += (Data[DataRowPtr[LayerIndex-1]+PrevNeuronIndex]) *
                            (Weights[WeightsRowPtr[LayerIndex] +
                                     (PrevLayerSize)*NeuronIndex +
                                     PrevNeuronIndex]);
                    }
                }
            
                Sum += c0 + c1 + c2 + c3 +
                    Weights[WeightsRowPtr[LayerIndex] +
                            (PrevLayerSize)*NeuronIndex +
                            LayerSizes[LayerIndex-1]];

                Data[DataRowPtr[LayerIndex]+NeuronIndex] = (1.0f/(1.0f+(r32)exp(-Sum)));
            }
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

/****************************************************************/
    
    {
        u32 LayerIndex = 0;
        for (;
             LayerIndex+3 < LayerSizes[(LayerCount)-1];
             LayerIndex+=4)
        {
            Delta[DataRowPtr[(LayerCount)-1]+LayerIndex] =
                Data[DataRowPtr[(LayerCount)-1]+LayerIndex] *
                (1 - Data[DataRowPtr[(LayerCount)-1]+LayerIndex]) *
                (Target[LayerIndex] - Data[DataRowPtr[(LayerCount)-1]+LayerIndex]);
            
            Delta[DataRowPtr[(LayerCount)-1]+LayerIndex+1] =
                Data[DataRowPtr[(LayerCount)-1]+LayerIndex+1] *
                (1 - Data[DataRowPtr[(LayerCount)-1]+LayerIndex+1]) *
                (Target[LayerIndex+1] - Data[DataRowPtr[(LayerCount)-1]+LayerIndex+1]);
            
            Delta[DataRowPtr[(LayerCount)-1]+LayerIndex+2] =
                Data[DataRowPtr[(LayerCount)-1]+LayerIndex+2] *
                (1 - Data[DataRowPtr[(LayerCount)-1]+LayerIndex+2]) *
                (Target[LayerIndex+2] - Data[DataRowPtr[(LayerCount)-1]+LayerIndex+2]);
            
            Delta[DataRowPtr[(LayerCount)-1]+LayerIndex+3] =
                Data[DataRowPtr[(LayerCount)-1]+LayerIndex+3] *
                (1 - Data[DataRowPtr[(LayerCount)-1]+LayerIndex+3]) *
                (Target[LayerIndex+3] - Data[DataRowPtr[(LayerCount)-1]+LayerIndex+3]);
        }
        for (;
             LayerIndex < LayerSizes[(LayerCount)-1];
             ++LayerIndex)
        {
            Delta[DataRowPtr[(LayerCount)-1]+LayerIndex] =
                Data[DataRowPtr[(LayerCount)-1]+LayerIndex] *
                (1 - Data[DataRowPtr[(LayerCount)-1]+LayerIndex]) *
                (Target[LayerIndex] - Data[DataRowPtr[(LayerCount)-1]+LayerIndex]);
        }
    }

/****************************************************************/
    
    for(u32 LayerIndex = LayerCount-2;
        LayerIndex > 0;
        --LayerIndex)
    {
        u32 CurrLayerSize = LayerSizes[LayerIndex]+1;
        while(CurrLayerSize&15)
        {
            ++CurrLayerSize;
        }
        {
            u32 NeuronIndex = 0;
            for(;
                NeuronIndex+3 < LayerSizes[LayerIndex];
                NeuronIndex+=4)
            {
                r32 Sum0 = 0.0f;
                r32 c00 = 0.0f;
                r32 c01 = 0.0f;
                r32 c02 = 0.0f;
                r32 c03 = 0.0f;

                r32 Sum1 = 0.0f;
                r32 c10 = 0.0f;
                r32 c11 = 0.0f;
                r32 c12 = 0.0f;
                r32 c13 = 0.0f;

                r32 Sum2 = 0.0f;
                r32 c20 = 0.0f;
                r32 c21 = 0.0f;
                r32 c22 = 0.0f;
                r32 c23 = 0.0f;

                r32 Sum3 = 0.0f;
                r32 c30 = 0.0f;
                r32 c31 = 0.0f;
                r32 c32 = 0.0f;
                r32 c33 = 0.0f;

                {
                    u32 NextNeuronIndex=0;
                    for(;
                        NextNeuronIndex+3 < LayerSizes[LayerIndex+1];
                        NextNeuronIndex+=4)
                    {
                        c00 += Delta[DataRowPtr[LayerIndex+1]+NextNeuronIndex] *
                            Weights[WeightsRowPtr[LayerIndex + 1] +
                                    (CurrLayerSize)*NextNeuronIndex +
                                    (NeuronIndex+0)];
                        c01 += Delta[DataRowPtr[LayerIndex+1]+NextNeuronIndex+1] *
                            Weights[WeightsRowPtr[LayerIndex + 1] +
                                    (CurrLayerSize)*(NextNeuronIndex+1) +
                                    (NeuronIndex+0)];
                        c02 += Delta[DataRowPtr[LayerIndex+1]+NextNeuronIndex+2] *
                            Weights[WeightsRowPtr[LayerIndex + 1] +
                                    (CurrLayerSize)*(NextNeuronIndex+2) +
                                    (NeuronIndex+0)];
                        c03 += Delta[DataRowPtr[LayerIndex+1]+NextNeuronIndex+3] *
                            Weights[WeightsRowPtr[LayerIndex + 1] +
                                    (CurrLayerSize)*(NextNeuronIndex+3) +
                                    (NeuronIndex+0)];
                    
                        c10 += Delta[DataRowPtr[LayerIndex+1]+NextNeuronIndex] *
                            Weights[WeightsRowPtr[LayerIndex + 1] +
                                    (CurrLayerSize)*NextNeuronIndex +
                                    (NeuronIndex+1)];
                        c11 += Delta[DataRowPtr[LayerIndex+1]+NextNeuronIndex+1] *
                            Weights[WeightsRowPtr[LayerIndex + 1] +
                                    (CurrLayerSize)*(NextNeuronIndex+1) +
                                    (NeuronIndex+1)];
                        c12 += Delta[DataRowPtr[LayerIndex+1]+NextNeuronIndex+2] *
                            Weights[WeightsRowPtr[LayerIndex + 1] +
                                    (CurrLayerSize)*(NextNeuronIndex+2) +
                                    (NeuronIndex+1)];
                        c13 += Delta[DataRowPtr[LayerIndex+1]+NextNeuronIndex+3] *
                            Weights[WeightsRowPtr[LayerIndex + 1] +
                                    (CurrLayerSize)*(NextNeuronIndex+3) +
                                    (NeuronIndex+1)];
                    
                        c20 += Delta[DataRowPtr[LayerIndex+1]+NextNeuronIndex] *
                            Weights[WeightsRowPtr[LayerIndex + 1] +
                                    (CurrLayerSize)*NextNeuronIndex +
                                    (NeuronIndex+2)];
                        c21 += Delta[DataRowPtr[LayerIndex+1]+NextNeuronIndex+1] *
                            Weights[WeightsRowPtr[LayerIndex + 1] +
                                    (CurrLayerSize)*(NextNeuronIndex+1) +
                                    (NeuronIndex+2)];
                        c22 += Delta[DataRowPtr[LayerIndex+1]+NextNeuronIndex+2] *
                            Weights[WeightsRowPtr[LayerIndex + 1] +
                                    (CurrLayerSize)*(NextNeuronIndex+2) +
                                    (NeuronIndex+2)];
                        c23 += Delta[DataRowPtr[LayerIndex+1]+NextNeuronIndex+3] *
                            Weights[WeightsRowPtr[LayerIndex + 1] +
                                    (CurrLayerSize)*(NextNeuronIndex+3) +
                                    (NeuronIndex+2)];
                    
                        c30 += Delta[DataRowPtr[LayerIndex+1]+NextNeuronIndex] *
                            Weights[WeightsRowPtr[LayerIndex + 1] +
                                    (CurrLayerSize)*NextNeuronIndex +
                                    (NeuronIndex+3)];
                        c31 += Delta[DataRowPtr[LayerIndex+1]+NextNeuronIndex+1] *
                            Weights[WeightsRowPtr[LayerIndex + 1] +
                                    (CurrLayerSize)*(NextNeuronIndex+1) +
                                    (NeuronIndex+3)];
                        c32 += Delta[DataRowPtr[LayerIndex+1]+NextNeuronIndex+2] *
                            Weights[WeightsRowPtr[LayerIndex + 1] +
                                    (CurrLayerSize)*(NextNeuronIndex+2) +
                                    (NeuronIndex+3)];
                        c33 += Delta[DataRowPtr[LayerIndex+1]+NextNeuronIndex+3] *
                            Weights[WeightsRowPtr[LayerIndex + 1] +
                                    (CurrLayerSize)*(NextNeuronIndex+3) +
                                    (NeuronIndex+3)];
                    }
                    for(;
                        NextNeuronIndex < LayerSizes[LayerIndex+1];
                        ++NextNeuronIndex)
                    {
                        c00 += Delta[DataRowPtr[LayerIndex+1]+NextNeuronIndex] *
                            Weights[WeightsRowPtr[LayerIndex + 1] +
                                    (CurrLayerSize)*NextNeuronIndex +
                                    (NeuronIndex+0)];
                        c10 += Delta[DataRowPtr[LayerIndex+1]+NextNeuronIndex] *
                            Weights[WeightsRowPtr[LayerIndex + 1] +
                                    (CurrLayerSize)*NextNeuronIndex +
                                    (NeuronIndex+1)];
                        c20 += Delta[DataRowPtr[LayerIndex+1]+NextNeuronIndex] *
                            Weights[WeightsRowPtr[LayerIndex + 1] +
                                    (CurrLayerSize)*NextNeuronIndex +
                                    (NeuronIndex+2)];
                        c30 += Delta[DataRowPtr[LayerIndex+1]+NextNeuronIndex] *
                            Weights[WeightsRowPtr[LayerIndex + 1] +
                                    (CurrLayerSize)*NextNeuronIndex +
                                    (NeuronIndex+3)];
                    }
                }

                Sum0 += c00 + c01 + c02 + c03;

                Delta[DataRowPtr[LayerIndex] + (NeuronIndex+0)] =
                    Data[DataRowPtr[LayerIndex] + (NeuronIndex+0)] *
                    (1 - Data[DataRowPtr[LayerIndex] + (NeuronIndex+0)]) *
                    Sum0;
                
                Sum1 += c10 + c11 + c12 + c13;

                Delta[DataRowPtr[LayerIndex] + (NeuronIndex+1)] =
                    Data[DataRowPtr[LayerIndex] + (NeuronIndex+1)] *
                    (1 - Data[DataRowPtr[LayerIndex] + (NeuronIndex+1)]) *
                    Sum1;
                
                Sum2 += c20 + c21 + c22 + c23;

                Delta[DataRowPtr[LayerIndex] + (NeuronIndex+2)] =
                    Data[DataRowPtr[LayerIndex] + (NeuronIndex+2)] *
                    (1 - Data[DataRowPtr[LayerIndex] + (NeuronIndex+2)]) *
                    Sum2;
                
                Sum3 += c30 + c31 + c32 + c33;

                Delta[DataRowPtr[LayerIndex] + (NeuronIndex+3)] =
                    Data[DataRowPtr[LayerIndex] + (NeuronIndex+3)] *
                    (1 - Data[DataRowPtr[LayerIndex] + (NeuronIndex+3)]) *
                    Sum3;
            }
            for(;
                NeuronIndex < LayerSizes[LayerIndex];
                ++NeuronIndex)
            {
                r32 Sum = 0.0f;
                r32 c0 = 0.0f;
                r32 c1 = 0.0f;
                r32 c2 = 0.0f;
                r32 c3 = 0.0f;

                {
                    u32 NextNeuronIndex=0;
                    for(;
                        NextNeuronIndex+3 < LayerSizes[LayerIndex+1];
                        NextNeuronIndex+=4)
                    {
                        c0 += Delta[DataRowPtr[LayerIndex+1]+NextNeuronIndex] *
                            Weights[WeightsRowPtr[LayerIndex + 1] +
                                    (CurrLayerSize)*NextNeuronIndex +
                                    NeuronIndex];
                        c1 += Delta[DataRowPtr[LayerIndex+1]+NextNeuronIndex+1] *
                            Weights[WeightsRowPtr[LayerIndex + 1] +
                                    (CurrLayerSize)*(NextNeuronIndex+1) +
                                    NeuronIndex];
                        c2 += Delta[DataRowPtr[LayerIndex+1]+NextNeuronIndex+2] *
                            Weights[WeightsRowPtr[LayerIndex + 1] +
                                    (CurrLayerSize)*(NextNeuronIndex+2) +
                                    NeuronIndex];
                        c3 += Delta[DataRowPtr[LayerIndex+1]+NextNeuronIndex+3] *
                            Weights[WeightsRowPtr[LayerIndex + 1] +
                                    (CurrLayerSize)*(NextNeuronIndex+3) +
                                    NeuronIndex];
                    }
                    for(;
                        NextNeuronIndex < LayerSizes[LayerIndex+1];
                        ++NextNeuronIndex)
                    {
                        c0 += Delta[DataRowPtr[LayerIndex+1]+NextNeuronIndex] *
                            Weights[WeightsRowPtr[LayerIndex + 1] +
                                    (CurrLayerSize)*NextNeuronIndex +
                                    NeuronIndex];
                    }
                }

                Sum += c0 + c1 + c2 + c3;

                Delta[DataRowPtr[LayerIndex] + NeuronIndex] =
                    Data[DataRowPtr[LayerIndex] + NeuronIndex] *
                    (1 - Data[DataRowPtr[LayerIndex] + NeuronIndex]) *
                    Sum;
            }
        }
    }

/****************************************************************/


	for(u32 LayerIndex = 1;
        LayerIndex < LayerCount;
        ++LayerIndex)
	{
        u32 PrevLayerSize = LayerSizes[LayerIndex-1]+1;
        while(PrevLayerSize&15)
        {
            ++PrevLayerSize;
        }

        {
            u32 NeuronIndex = 0;
            for(;
                NeuronIndex+3 < LayerSizes[LayerIndex];
                NeuronIndex+=4)
            {
                {
                    u32 PrevNeuronIndex = 0;
                    for(;
                        PrevNeuronIndex+3 < LayerSizes[LayerIndex-1];
                        PrevNeuronIndex+=4)
                    {
                        
                        Weights[WeightsRowPtr[LayerIndex] + PrevLayerSize * (NeuronIndex+0) + PrevNeuronIndex] +=
                            (Alpha) * WeightDelta[WeightsRowPtr[LayerIndex] +
                                                  PrevLayerSize * (NeuronIndex+0) + PrevNeuronIndex];
                        Weights[WeightsRowPtr[LayerIndex] + PrevLayerSize * (NeuronIndex+0) + PrevNeuronIndex + 1] +=
                            (Alpha) * WeightDelta[WeightsRowPtr[LayerIndex] +
                                                  PrevLayerSize * (NeuronIndex+0) + PrevNeuronIndex + 1];
                        Weights[WeightsRowPtr[LayerIndex] + PrevLayerSize * (NeuronIndex+0) + PrevNeuronIndex + 2] +=
                            (Alpha) * WeightDelta[WeightsRowPtr[LayerIndex] +
                                                  PrevLayerSize * (NeuronIndex+0) + PrevNeuronIndex + 2];
                        Weights[WeightsRowPtr[LayerIndex] + PrevLayerSize * (NeuronIndex+0) + PrevNeuronIndex + 3] +=
                            (Alpha) * WeightDelta[WeightsRowPtr[LayerIndex] +
                                                  PrevLayerSize * (NeuronIndex+0) + PrevNeuronIndex + 3];
                        
                        Weights[WeightsRowPtr[LayerIndex] + PrevLayerSize * (NeuronIndex+1) + PrevNeuronIndex] +=
                            (Alpha) * WeightDelta[WeightsRowPtr[LayerIndex] +
                                                  PrevLayerSize * (NeuronIndex+1) + PrevNeuronIndex];
                        Weights[WeightsRowPtr[LayerIndex] + PrevLayerSize * (NeuronIndex+1) + PrevNeuronIndex + 1] +=
                            (Alpha) * WeightDelta[WeightsRowPtr[LayerIndex] +
                                                  PrevLayerSize * (NeuronIndex+1) + PrevNeuronIndex + 1];
                        Weights[WeightsRowPtr[LayerIndex] + PrevLayerSize * (NeuronIndex+1) + PrevNeuronIndex + 2] +=
                            (Alpha) * WeightDelta[WeightsRowPtr[LayerIndex] +
                                                  PrevLayerSize * (NeuronIndex+1) + PrevNeuronIndex + 2];
                        Weights[WeightsRowPtr[LayerIndex] + PrevLayerSize * (NeuronIndex+1) + PrevNeuronIndex + 3] +=
                            (Alpha) * WeightDelta[WeightsRowPtr[LayerIndex] +
                                                  PrevLayerSize * (NeuronIndex+1) + PrevNeuronIndex + 3];
                        
                        Weights[WeightsRowPtr[LayerIndex] + PrevLayerSize * (NeuronIndex+2) + PrevNeuronIndex] +=
                            (Alpha) * WeightDelta[WeightsRowPtr[LayerIndex] +
                                                  PrevLayerSize * (NeuronIndex+2) + PrevNeuronIndex];
                        Weights[WeightsRowPtr[LayerIndex] + PrevLayerSize * (NeuronIndex+2) + PrevNeuronIndex + 1] +=
                            (Alpha) * WeightDelta[WeightsRowPtr[LayerIndex] +
                                                  PrevLayerSize * (NeuronIndex+2) + PrevNeuronIndex + 1];
                        Weights[WeightsRowPtr[LayerIndex] + PrevLayerSize * (NeuronIndex+2) + PrevNeuronIndex + 2] +=
                            (Alpha) * WeightDelta[WeightsRowPtr[LayerIndex] +
                                                  PrevLayerSize * (NeuronIndex+2) + PrevNeuronIndex + 2];
                        Weights[WeightsRowPtr[LayerIndex] + PrevLayerSize * (NeuronIndex+2) + PrevNeuronIndex + 3] +=
                            (Alpha) * WeightDelta[WeightsRowPtr[LayerIndex] +
                                                  PrevLayerSize * (NeuronIndex+2) + PrevNeuronIndex + 3];
                        
                        Weights[WeightsRowPtr[LayerIndex] + PrevLayerSize * (NeuronIndex+3) + PrevNeuronIndex] +=
                            (Alpha) * WeightDelta[WeightsRowPtr[LayerIndex] +
                                                  PrevLayerSize * (NeuronIndex+3) + PrevNeuronIndex];
                        Weights[WeightsRowPtr[LayerIndex] + PrevLayerSize * (NeuronIndex+3) + PrevNeuronIndex + 1] +=
                            (Alpha) * WeightDelta[WeightsRowPtr[LayerIndex] +
                                                  PrevLayerSize * (NeuronIndex+3) + PrevNeuronIndex + 1];
                        Weights[WeightsRowPtr[LayerIndex] + PrevLayerSize * (NeuronIndex+3) + PrevNeuronIndex + 2] +=
                            (Alpha) * WeightDelta[WeightsRowPtr[LayerIndex] +
                                                  PrevLayerSize * (NeuronIndex+3) + PrevNeuronIndex + 2];
                        Weights[WeightsRowPtr[LayerIndex] + PrevLayerSize * (NeuronIndex+3) + PrevNeuronIndex + 3] +=
                            (Alpha) * WeightDelta[WeightsRowPtr[LayerIndex] +
                                                  PrevLayerSize * (NeuronIndex+3) + PrevNeuronIndex + 3];
                    }
                    for(;
                        PrevNeuronIndex < LayerSizes[LayerIndex-1];
                        ++PrevNeuronIndex)
                    {
                        Weights[WeightsRowPtr[LayerIndex] + PrevLayerSize * (NeuronIndex+0) + PrevNeuronIndex] +=
                            (Alpha) * WeightDelta[WeightsRowPtr[LayerIndex] +
                                                  PrevLayerSize * (NeuronIndex+0) + PrevNeuronIndex];
                    
                        Weights[WeightsRowPtr[LayerIndex] + PrevLayerSize * (NeuronIndex+1) + PrevNeuronIndex] +=
                            (Alpha) * WeightDelta[WeightsRowPtr[LayerIndex] +
                                                  PrevLayerSize * (NeuronIndex+1) + PrevNeuronIndex];
                    
                        Weights[WeightsRowPtr[LayerIndex] + PrevLayerSize * (NeuronIndex+2) + PrevNeuronIndex] +=
                            (Alpha) * WeightDelta[WeightsRowPtr[LayerIndex] +
                                                  PrevLayerSize * (NeuronIndex+2) + PrevNeuronIndex];
                    
                        Weights[WeightsRowPtr[LayerIndex] + PrevLayerSize * (NeuronIndex+3) + PrevNeuronIndex] +=
                            (Alpha) * WeightDelta[WeightsRowPtr[LayerIndex] +
                                                  PrevLayerSize * (NeuronIndex+3) + PrevNeuronIndex];
                    }
                }

                Weights[WeightsRowPtr[LayerIndex] + PrevLayerSize * (NeuronIndex+0) + LayerSizes[LayerIndex-1]] +=
                    Alpha * WeightDelta[WeightsRowPtr[LayerIndex] +
                                        PrevLayerSize * (NeuronIndex+0) + LayerSizes[LayerIndex-1]];
                
                Weights[WeightsRowPtr[LayerIndex] + PrevLayerSize * (NeuronIndex+1) + LayerSizes[LayerIndex-1]] +=
                    Alpha * WeightDelta[WeightsRowPtr[LayerIndex] +
                                        PrevLayerSize * (NeuronIndex+1) + LayerSizes[LayerIndex-1]];
                
                Weights[WeightsRowPtr[LayerIndex] + PrevLayerSize * (NeuronIndex+2) + LayerSizes[LayerIndex-1]] +=
                    Alpha * WeightDelta[WeightsRowPtr[LayerIndex] +
                                        PrevLayerSize * (NeuronIndex+2) + LayerSizes[LayerIndex-1]];
                
                Weights[WeightsRowPtr[LayerIndex] + PrevLayerSize * (NeuronIndex+3) + LayerSizes[LayerIndex-1]] +=
                    Alpha * WeightDelta[WeightsRowPtr[LayerIndex] +
                                        PrevLayerSize * (NeuronIndex+3) + LayerSizes[LayerIndex-1]];
            }
        
            for(;
                NeuronIndex < LayerSizes[LayerIndex];
                ++NeuronIndex)
            {
                {
                    u32 PrevNeuronIndex = 0;
                    for(;
                        PrevNeuronIndex+3 < LayerSizes[LayerIndex-1];
                        PrevNeuronIndex+=4)
                    {
                        Weights[WeightsRowPtr[LayerIndex] + PrevLayerSize * NeuronIndex + PrevNeuronIndex] +=
                            (Alpha) * WeightDelta[WeightsRowPtr[LayerIndex] +
                                                  PrevLayerSize * NeuronIndex + PrevNeuronIndex];
                        Weights[WeightsRowPtr[LayerIndex] + PrevLayerSize * NeuronIndex + PrevNeuronIndex + 1] +=
                            (Alpha) * WeightDelta[WeightsRowPtr[LayerIndex] +
                                                  PrevLayerSize * NeuronIndex + PrevNeuronIndex + 1];
                        Weights[WeightsRowPtr[LayerIndex] + PrevLayerSize * NeuronIndex + PrevNeuronIndex + 2] +=
                            (Alpha) * WeightDelta[WeightsRowPtr[LayerIndex] +
                                                  PrevLayerSize * NeuronIndex + PrevNeuronIndex + 2];
                        Weights[WeightsRowPtr[LayerIndex] + PrevLayerSize * NeuronIndex + PrevNeuronIndex + 3] +=
                            (Alpha) * WeightDelta[WeightsRowPtr[LayerIndex] +
                                                  PrevLayerSize * NeuronIndex + PrevNeuronIndex + 3];
                    }
                    for(;
                        PrevNeuronIndex < LayerSizes[LayerIndex-1];
                        ++PrevNeuronIndex)
                    {
                        Weights[WeightsRowPtr[LayerIndex] + PrevLayerSize * NeuronIndex + PrevNeuronIndex] +=
                            (Alpha) * WeightDelta[WeightsRowPtr[LayerIndex] +
                                                  PrevLayerSize * NeuronIndex + PrevNeuronIndex];
                    }
                }

                Weights[WeightsRowPtr[LayerIndex] + PrevLayerSize * NeuronIndex + LayerSizes[LayerIndex-1]] +=
                    Alpha * WeightDelta[WeightsRowPtr[LayerIndex] +
                                        PrevLayerSize * NeuronIndex + LayerSizes[LayerIndex-1]];
            }
        }
	}

/****************************************************************/
    
	for(u32 LayerIndex = 1;
        LayerIndex < LayerCount;
        ++LayerIndex)
	{
        u32 PrevLayerSize = LayerSizes[LayerIndex-1]+1;
        while(PrevLayerSize&15)
        {
            ++PrevLayerSize;
        }

        {
            u32 NeuronIndex = 0;
        
            for(;
                NeuronIndex+3 < LayerSizes[LayerIndex];
                NeuronIndex+=4)
            {
                {
                    u32 PrevNeuronIndex = 0;
                    for(;
                        PrevNeuronIndex+3 < LayerSizes[LayerIndex-1];
                        PrevNeuronIndex+=4)
                    {
                        
                        WeightDelta[WeightsRowPtr[LayerIndex] +
                                    (PrevLayerSize * NeuronIndex)
                                    + (PrevNeuronIndex+0)] =
                            Beta * Delta[DataRowPtr[LayerIndex]+NeuronIndex] *
                            Data[DataRowPtr[LayerIndex-1] + (PrevNeuronIndex+0)];
                
                        WeightDelta[WeightsRowPtr[LayerIndex] +
                                    (PrevLayerSize * NeuronIndex)
                                    + (PrevNeuronIndex+1)] =
                            Beta * Delta[DataRowPtr[LayerIndex]+NeuronIndex] *
                            Data[DataRowPtr[LayerIndex-1] + (PrevNeuronIndex+1)];
                
                        WeightDelta[WeightsRowPtr[LayerIndex] +
                                    (PrevLayerSize * NeuronIndex)
                                    + (PrevNeuronIndex+2)] =
                            Beta * Delta[DataRowPtr[LayerIndex]+NeuronIndex] *
                            Data[DataRowPtr[LayerIndex-1] + (PrevNeuronIndex+2)];
                
                        WeightDelta[WeightsRowPtr[LayerIndex] +
                                    (PrevLayerSize * NeuronIndex)
                                    + (PrevNeuronIndex+3)] =
                            Beta * Delta[DataRowPtr[LayerIndex]+NeuronIndex] *
                            Data[DataRowPtr[LayerIndex-1] + (PrevNeuronIndex+3)];
                
                        Weights[WeightsRowPtr[LayerIndex] +
                                (PrevLayerSize * NeuronIndex) +
                                (PrevNeuronIndex+0)] +=
                            WeightDelta[WeightsRowPtr[LayerIndex] +
                                        PrevLayerSize * NeuronIndex + (PrevNeuronIndex+0)];
                
                        Weights[WeightsRowPtr[LayerIndex] +
                                (PrevLayerSize * NeuronIndex) +
                                (PrevNeuronIndex+1)] +=
                            WeightDelta[WeightsRowPtr[LayerIndex] +
                                        PrevLayerSize * NeuronIndex + (PrevNeuronIndex+1)];
                
                        Weights[WeightsRowPtr[LayerIndex] +
                                (PrevLayerSize * NeuronIndex) +
                                (PrevNeuronIndex+2)] +=
                            WeightDelta[WeightsRowPtr[LayerIndex] +
                                        PrevLayerSize * NeuronIndex + (PrevNeuronIndex+2)];
                
                        Weights[WeightsRowPtr[LayerIndex] +
                                (PrevLayerSize * NeuronIndex) +
                                (PrevNeuronIndex+3)] +=
                            WeightDelta[WeightsRowPtr[LayerIndex] +
                                        PrevLayerSize * NeuronIndex + (PrevNeuronIndex+3)];


                        
                        WeightDelta[WeightsRowPtr[LayerIndex] +
                                    (PrevLayerSize * (NeuronIndex+1))
                                    + (PrevNeuronIndex+0)] =
                            Beta * Delta[DataRowPtr[LayerIndex]+(NeuronIndex+1)] *
                            Data[DataRowPtr[LayerIndex-1] + (PrevNeuronIndex+0)];
                
                        WeightDelta[WeightsRowPtr[LayerIndex] +
                                    (PrevLayerSize * (NeuronIndex+1))
                                    + (PrevNeuronIndex+1)] =
                            Beta * Delta[DataRowPtr[LayerIndex]+(NeuronIndex+1)] *
                            Data[DataRowPtr[LayerIndex-1] + (PrevNeuronIndex+1)];
                
                        WeightDelta[WeightsRowPtr[LayerIndex] +
                                    (PrevLayerSize * (NeuronIndex+1))
                                    + (PrevNeuronIndex+2)] =
                            Beta * Delta[DataRowPtr[LayerIndex]+(NeuronIndex+1)] *
                            Data[DataRowPtr[LayerIndex-1] + (PrevNeuronIndex+2)];
                
                        WeightDelta[WeightsRowPtr[LayerIndex] +
                                    (PrevLayerSize * (NeuronIndex+1))
                                    + (PrevNeuronIndex+3)] =
                            Beta * Delta[DataRowPtr[LayerIndex]+(NeuronIndex+1)] *
                            Data[DataRowPtr[LayerIndex-1] + (PrevNeuronIndex+3)];
                
                        Weights[WeightsRowPtr[LayerIndex] +
                                (PrevLayerSize * (NeuronIndex+1)) +
                                (PrevNeuronIndex+0)] +=
                            WeightDelta[WeightsRowPtr[LayerIndex] +
                                        PrevLayerSize * (NeuronIndex+1) + (PrevNeuronIndex+0)];
                
                        Weights[WeightsRowPtr[LayerIndex] +
                                (PrevLayerSize * (NeuronIndex+1)) +
                                (PrevNeuronIndex+1)] +=
                            WeightDelta[WeightsRowPtr[LayerIndex] +
                                        PrevLayerSize * (NeuronIndex+1) + (PrevNeuronIndex+1)];
                
                        Weights[WeightsRowPtr[LayerIndex] +
                                (PrevLayerSize * (NeuronIndex+1)) +
                                (PrevNeuronIndex+2)] +=
                            WeightDelta[WeightsRowPtr[LayerIndex] +
                                        PrevLayerSize * (NeuronIndex+1) + (PrevNeuronIndex+2)];
                
                        Weights[WeightsRowPtr[LayerIndex] +
                                (PrevLayerSize * (NeuronIndex+1)) +
                                (PrevNeuronIndex+3)] +=
                            WeightDelta[WeightsRowPtr[LayerIndex] +
                                        PrevLayerSize * (NeuronIndex+1) + (PrevNeuronIndex+3)];



                        
                        WeightDelta[WeightsRowPtr[LayerIndex] +
                                    (PrevLayerSize * (NeuronIndex+2))
                                    + (PrevNeuronIndex+0)] =
                            Beta * Delta[DataRowPtr[LayerIndex]+(NeuronIndex+2)] *
                            Data[DataRowPtr[LayerIndex-1] + (PrevNeuronIndex+0)];
                
                        WeightDelta[WeightsRowPtr[LayerIndex] +
                                    (PrevLayerSize * (NeuronIndex+2))
                                    + (PrevNeuronIndex+1)] =
                            Beta * Delta[DataRowPtr[LayerIndex]+(NeuronIndex+2)] *
                            Data[DataRowPtr[LayerIndex-1] + (PrevNeuronIndex+1)];
                
                        WeightDelta[WeightsRowPtr[LayerIndex] +
                                    (PrevLayerSize * (NeuronIndex+2))
                                    + (PrevNeuronIndex+2)] =
                            Beta * Delta[DataRowPtr[LayerIndex]+(NeuronIndex+2)] *
                            Data[DataRowPtr[LayerIndex-1] + (PrevNeuronIndex+2)];
                
                        WeightDelta[WeightsRowPtr[LayerIndex] +
                                    (PrevLayerSize * (NeuronIndex+2))
                                    + (PrevNeuronIndex+3)] =
                            Beta * Delta[DataRowPtr[LayerIndex]+(NeuronIndex+2)] *
                            Data[DataRowPtr[LayerIndex-1] + (PrevNeuronIndex+3)];
                
                        Weights[WeightsRowPtr[LayerIndex] +
                                (PrevLayerSize * (NeuronIndex+2)) +
                                (PrevNeuronIndex+0)] +=
                            WeightDelta[WeightsRowPtr[LayerIndex] +
                                        PrevLayerSize * (NeuronIndex+2) + (PrevNeuronIndex+0)];
                
                        Weights[WeightsRowPtr[LayerIndex] +
                                (PrevLayerSize * (NeuronIndex+2)) +
                                (PrevNeuronIndex+1)] +=
                            WeightDelta[WeightsRowPtr[LayerIndex] +
                                        PrevLayerSize * (NeuronIndex+2) + (PrevNeuronIndex+1)];
                
                        Weights[WeightsRowPtr[LayerIndex] +
                                (PrevLayerSize * (NeuronIndex+2)) +
                                (PrevNeuronIndex+2)] +=
                            WeightDelta[WeightsRowPtr[LayerIndex] +
                                        PrevLayerSize * (NeuronIndex+2) + (PrevNeuronIndex+2)];
                
                        Weights[WeightsRowPtr[LayerIndex] +
                                (PrevLayerSize * (NeuronIndex+2)) +
                                (PrevNeuronIndex+3)] +=
                            WeightDelta[WeightsRowPtr[LayerIndex] +
                                        PrevLayerSize * (NeuronIndex+2) + (PrevNeuronIndex+3)];


                        
                        WeightDelta[WeightsRowPtr[LayerIndex] +
                                    (PrevLayerSize * (NeuronIndex+3))
                                    + (PrevNeuronIndex+0)] =
                            Beta * Delta[DataRowPtr[LayerIndex]+(NeuronIndex+3)] *
                            Data[DataRowPtr[LayerIndex-1] + (PrevNeuronIndex+0)];
                
                        WeightDelta[WeightsRowPtr[LayerIndex] +
                                    (PrevLayerSize * (NeuronIndex+3))
                                    + (PrevNeuronIndex+1)] =
                            Beta * Delta[DataRowPtr[LayerIndex]+(NeuronIndex+3)] *
                            Data[DataRowPtr[LayerIndex-1] + (PrevNeuronIndex+1)];
                
                        WeightDelta[WeightsRowPtr[LayerIndex] +
                                    (PrevLayerSize * (NeuronIndex+3))
                                    + (PrevNeuronIndex+2)] =
                            Beta * Delta[DataRowPtr[LayerIndex]+(NeuronIndex+3)] *
                            Data[DataRowPtr[LayerIndex-1] + (PrevNeuronIndex+2)];
                
                        WeightDelta[WeightsRowPtr[LayerIndex] +
                                    (PrevLayerSize * (NeuronIndex+3))
                                    + (PrevNeuronIndex+3)] =
                            Beta * Delta[DataRowPtr[LayerIndex]+(NeuronIndex+3)] *
                            Data[DataRowPtr[LayerIndex-1] + (PrevNeuronIndex+3)];
                
                        Weights[WeightsRowPtr[LayerIndex] +
                                (PrevLayerSize * (NeuronIndex+3)) +
                                (PrevNeuronIndex+0)] +=
                            WeightDelta[WeightsRowPtr[LayerIndex] +
                                        PrevLayerSize * (NeuronIndex+3) + (PrevNeuronIndex+0)];
                
                        Weights[WeightsRowPtr[LayerIndex] +
                                (PrevLayerSize * (NeuronIndex+3)) +
                                (PrevNeuronIndex+1)] +=
                            WeightDelta[WeightsRowPtr[LayerIndex] +
                                        PrevLayerSize * (NeuronIndex+3) + (PrevNeuronIndex+1)];
                
                        Weights[WeightsRowPtr[LayerIndex] +
                                (PrevLayerSize * (NeuronIndex+3)) +
                                (PrevNeuronIndex+2)] +=
                            WeightDelta[WeightsRowPtr[LayerIndex] +
                                        PrevLayerSize * (NeuronIndex+3) + (PrevNeuronIndex+2)];
                
                        Weights[WeightsRowPtr[LayerIndex] +
                                (PrevLayerSize * (NeuronIndex+3)) +
                                (PrevNeuronIndex+3)] +=
                            WeightDelta[WeightsRowPtr[LayerIndex] +
                                        PrevLayerSize * (NeuronIndex+3) + (PrevNeuronIndex+3)];

                    }
                    for(;
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
                        
                        
                        WeightDelta[WeightsRowPtr[LayerIndex] +
                                    (PrevLayerSize * (NeuronIndex+1))
                                    + PrevNeuronIndex] =
                            Beta * Delta[DataRowPtr[LayerIndex]+(NeuronIndex+1)] *
                            Data[DataRowPtr[LayerIndex-1] + PrevNeuronIndex];
                
                        Weights[WeightsRowPtr[LayerIndex] +
                                (PrevLayerSize * (NeuronIndex+1)) +
                                PrevNeuronIndex] +=
                            WeightDelta[WeightsRowPtr[LayerIndex] +
                                        PrevLayerSize * (NeuronIndex+1) + PrevNeuronIndex];
                        
                        
                        WeightDelta[WeightsRowPtr[LayerIndex] +
                                    (PrevLayerSize * (NeuronIndex+2))
                                    + PrevNeuronIndex] =
                            Beta * Delta[DataRowPtr[LayerIndex]+(NeuronIndex+2)] *
                            Data[DataRowPtr[LayerIndex-1] + PrevNeuronIndex];
                
                        Weights[WeightsRowPtr[LayerIndex] +
                                (PrevLayerSize * (NeuronIndex+2)) +
                                PrevNeuronIndex] +=
                            WeightDelta[WeightsRowPtr[LayerIndex] +
                                        PrevLayerSize * (NeuronIndex+2) + PrevNeuronIndex];
                        
                        
                        WeightDelta[WeightsRowPtr[LayerIndex] +
                                    (PrevLayerSize * (NeuronIndex+3))
                                    + PrevNeuronIndex] =
                            Beta * Delta[DataRowPtr[LayerIndex]+(NeuronIndex+3)] *
                            Data[DataRowPtr[LayerIndex-1] + PrevNeuronIndex];
                
                        Weights[WeightsRowPtr[LayerIndex] +
                                (PrevLayerSize * (NeuronIndex+3)) +
                                PrevNeuronIndex] +=
                            WeightDelta[WeightsRowPtr[LayerIndex] +
                                        PrevLayerSize * (NeuronIndex+3) + PrevNeuronIndex];
                        
                    }
                }

                                        
                WeightDelta[WeightsRowPtr[LayerIndex] +
                            (PrevLayerSize * (NeuronIndex+0))
                            + LayerSizes[LayerIndex-1]] =
                    Beta * Delta[DataRowPtr[LayerIndex]+(NeuronIndex+0)];
            
                Weights[WeightsRowPtr[LayerIndex] +
                        (PrevLayerSize * (NeuronIndex+0)) +
                        LayerSizes[LayerIndex-1]] +=
                    WeightDelta[WeightsRowPtr[LayerIndex] +
                                PrevLayerSize * (NeuronIndex+0) + LayerSizes[LayerIndex-1]];

                                        
                WeightDelta[WeightsRowPtr[LayerIndex] +
                            (PrevLayerSize * (NeuronIndex+1))
                            + LayerSizes[LayerIndex-1]] =
                    Beta * Delta[DataRowPtr[LayerIndex]+(NeuronIndex+1)];
            
                Weights[WeightsRowPtr[LayerIndex] +
                        (PrevLayerSize * (NeuronIndex+1)) +
                        LayerSizes[LayerIndex-1]] +=
                    WeightDelta[WeightsRowPtr[LayerIndex] +
                                PrevLayerSize * (NeuronIndex+1) + LayerSizes[LayerIndex-1]];

                                        
                WeightDelta[WeightsRowPtr[LayerIndex] +
                            (PrevLayerSize * (NeuronIndex+2))
                            + LayerSizes[LayerIndex-1]] =
                    Beta * Delta[DataRowPtr[LayerIndex]+(NeuronIndex+2)];
            
                Weights[WeightsRowPtr[LayerIndex] +
                        (PrevLayerSize * (NeuronIndex+2)) +
                        LayerSizes[LayerIndex-1]] +=
                    WeightDelta[WeightsRowPtr[LayerIndex] +
                                PrevLayerSize * (NeuronIndex+2) + LayerSizes[LayerIndex-1]];

                                        
                WeightDelta[WeightsRowPtr[LayerIndex] +
                            (PrevLayerSize * (NeuronIndex+3))
                            + LayerSizes[LayerIndex-1]] =
                    Beta * Delta[DataRowPtr[LayerIndex]+(NeuronIndex+3)];
            
                Weights[WeightsRowPtr[LayerIndex] +
                        (PrevLayerSize * (NeuronIndex+3)) +
                        LayerSizes[LayerIndex-1]] +=
                    WeightDelta[WeightsRowPtr[LayerIndex] +
                                PrevLayerSize * (NeuronIndex+3) + LayerSizes[LayerIndex-1]];
            }

            
            for(;
                NeuronIndex < LayerSizes[LayerIndex];
                ++NeuronIndex)
            {
                {
                    u32 PrevNeuronIndex = 0;
                    for(;
                        PrevNeuronIndex+3 < LayerSizes[LayerIndex-1];
                        PrevNeuronIndex+=4)
                    {
                        WeightDelta[WeightsRowPtr[LayerIndex] +
                                    (PrevLayerSize * NeuronIndex)
                                    + (PrevNeuronIndex+0)] =
                            Beta * Delta[DataRowPtr[LayerIndex]+NeuronIndex] *
                            Data[DataRowPtr[LayerIndex-1] + (PrevNeuronIndex+0)];
                
                        WeightDelta[WeightsRowPtr[LayerIndex] +
                                    (PrevLayerSize * NeuronIndex)
                                    + (PrevNeuronIndex+1)] =
                            Beta * Delta[DataRowPtr[LayerIndex]+NeuronIndex] *
                            Data[DataRowPtr[LayerIndex-1] + (PrevNeuronIndex+1)];
                
                        WeightDelta[WeightsRowPtr[LayerIndex] +
                                    (PrevLayerSize * NeuronIndex)
                                    + (PrevNeuronIndex+2)] =
                            Beta * Delta[DataRowPtr[LayerIndex]+NeuronIndex] *
                            Data[DataRowPtr[LayerIndex-1] + (PrevNeuronIndex+2)];
                
                        WeightDelta[WeightsRowPtr[LayerIndex] +
                                    (PrevLayerSize * NeuronIndex)
                                    + (PrevNeuronIndex+3)] =
                            Beta * Delta[DataRowPtr[LayerIndex]+NeuronIndex] *
                            Data[DataRowPtr[LayerIndex-1] + (PrevNeuronIndex+3)];
                
                        Weights[WeightsRowPtr[LayerIndex] +
                                (PrevLayerSize * NeuronIndex) +
                                (PrevNeuronIndex+0)] +=
                            WeightDelta[WeightsRowPtr[LayerIndex] +
                                        PrevLayerSize * NeuronIndex + (PrevNeuronIndex+0)];
                
                        Weights[WeightsRowPtr[LayerIndex] +
                                (PrevLayerSize * NeuronIndex) +
                                (PrevNeuronIndex+1)] +=
                            WeightDelta[WeightsRowPtr[LayerIndex] +
                                        PrevLayerSize * NeuronIndex + (PrevNeuronIndex+1)];
                
                        Weights[WeightsRowPtr[LayerIndex] +
                                (PrevLayerSize * NeuronIndex) +
                                (PrevNeuronIndex+2)] +=
                            WeightDelta[WeightsRowPtr[LayerIndex] +
                                        PrevLayerSize * NeuronIndex + (PrevNeuronIndex+2)];
                
                        Weights[WeightsRowPtr[LayerIndex] +
                                (PrevLayerSize * NeuronIndex) +
                                (PrevNeuronIndex+3)] +=
                            WeightDelta[WeightsRowPtr[LayerIndex] +
                                        PrevLayerSize * NeuronIndex + (PrevNeuronIndex+3)];
                    }
                    
                    for(;
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
}
