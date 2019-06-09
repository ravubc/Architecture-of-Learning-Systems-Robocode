public class BackpropogationAlgorithm {
	double [] X = new double [2];	
	double myError=1.0;
 double finalError=0.05;
 int epoch=0;
 public double input_Bias=1.0;
 public double hidden_Bias=1.0;
 public double [][]inputData= {{0,0},{0,1},{1,0},{1,1}};
 double []expectedOutput= {0,1,1,0};

//double [][]inputData= {{-1,-1},{-1,1},{1,-1},{1,1}};
//double []expectedOutput= {-1,1,1,-1};

 int number_Inputs=2;
 int number_Hidden=4;
 int number_Outputs=1;
 double[] outputError=new double[number_Outputs];
 double[] finalHiddenInput=new double[number_Hidden];
 double[] finalHiddenOutput=new double[number_Hidden];
 double[] finalOutput= new double[number_Outputs];
 boolean activationFunction=true;  //bipolar if true and binary sigmoid if false
 double[] hiddenEror=new double[number_Hidden];
 
 double [][]oldWeightInputToHidden=new double[number_Hidden][number_Inputs];   //last weight adjustments between input and hidden
 double [][]oldWeightHiddenToOutput=new double[number_Hidden][number_Outputs]; // last weight adjustments between hidden and output
 double [][]newWeightInputToHidden=new double[number_Hidden][number_Inputs];    // current weight adjustments between input and hidden
 double [][]newWeightHiddenToOutput=new double[number_Hidden][number_Outputs];  // current weight adjustments between hidden and output
 double[] outIn = new double [number_Hidden]; 


 double[] in=new double[number_Inputs];
  
 public void backPropagation()
 {
//System.out.println(finalError);
 Neuron n=new Neuron();
 n.calculateRandomWeight();
 
 ActivationFunction aF=new ActivationFunction(); 
 n.calculateRandomWeight();  

 while (myError>finalError && epoch<10000)
 {   //check if the error is less than 0.05 and epochs are less than 10000
	 myError = 0;
	 double weightSumInputHidden=0.0;
	 double weightSumHiddenOutput=0.0;
	 
	 epoch++;
	 
	 for(int nInputs=0;nInputs<4;nInputs++)
	 {
		 X[0] = inputData[nInputs][0];
		 X[1] = inputData[nInputs][1];
		 for(int i=0;i<number_Hidden;i++) 
		 {
			 for(int j=0;j<number_Inputs;j++) 
			 {
				 weightSumInputHidden= weightSumInputHidden+ (X[j]* n.weightInputToHidden[j][i]);			 
			 }
			 finalHiddenInput[i]= weightSumInputHidden + (n.weightBiasHidden[i] * input_Bias);
			 weightSumInputHidden=0;
			 
			 // activation function will be applied according to the value whether for bipolar or binary
			 if(activationFunction==false) 	
			 	{
				 	finalHiddenOutput[i]= aF.binarySigmoid(finalHiddenInput[i]); 
			 	}
			 else 
			 	{
				 	finalHiddenOutput[i]= aF.bipolarSigmoid(finalHiddenInput[i]);
			 	}
		 }
		 
		 //calculating weights from hidden to output
		 
		 for (int i=0;i<number_Outputs;i++)
		 {
		 	for (int j=0;j<number_Hidden;j++)	
		 	{
		 		weightSumHiddenOutput += (finalHiddenOutput[j] * n.weightHiddenToOutput[j][i]);
		 	}
		 	outIn[i] = weightSumHiddenOutput+(hidden_Bias * n.weightBiasOutput[0]);
		 	if (activationFunction == true) 
		 	{
		 		finalOutput[i] = aF.binarySigmoid(outIn[i]);
		 	} 
		 	else 
		 	{
		 		finalOutput[i] = aF.bipolarSigmoid(outIn[i]);
		 	}
		 	//System.out.println(finalOutput[i]);
		 	weightSumHiddenOutput=0;
		 }

		
		 
		 double errorChange= (Math.pow((expectedOutput[nInputs]-finalOutput[0]), 2))*0.5; 
		 myError= myError+ errorChange;
		 //System.out.println(expectedOutput[nInputs]);
		 //System.out.println(errorChange);
		 
		 
		 // error at output
		 for (int i=0; i<number_Outputs; i++)
		 {
			 if (activationFunction == false) 
			 {
				 outputError[i] = (expectedOutput[nInputs] - finalOutput[i])*(aF.derivativeBinarySigmoid(outIn[i]));
			 }
			 else 
			 {
				 outputError[i] = (expectedOutput[nInputs] - finalOutput[i])*(aF.derivativeBipolarSigmoid(outIn[i]));
			 }
			 
			 for (int k=0; k<number_Hidden; k++)
			 {
				 oldWeightHiddenToOutput[k][i] = newWeightHiddenToOutput[k][i];
				 newWeightHiddenToOutput[k][i] = (outputError[i] * finalHiddenOutput[k] * n.learningRate) + (oldWeightHiddenToOutput[k][i]* n.momemtum);
			 }
			 	n.weightBiasOutput[i] += n.learningRate * outputError[i];
		 
		 }
		 
		 for (int k=0; k<number_Hidden; k++)
		 {
			 for (int i=0; i<number_Outputs; i++)
			 {
				 weightSumHiddenOutput += outputError[i]* n.weightHiddenToOutput[k][i];
			 }
		 
			 if (activationFunction == false) 
			 {
				 hiddenEror[k] = weightSumHiddenOutput * aF.derivativeBinarySigmoid(finalHiddenInput[k]);
			 } 
			 else 
			 {
				 hiddenEror[k] = weightSumHiddenOutput * aF.derivativeBipolarSigmoid(finalHiddenInput[k]);
			 }
			 
			 
			 for (int g=0; g<number_Inputs;g++) 
			 {
				 oldWeightInputToHidden[k][g] = newWeightInputToHidden[k][g];
				 newWeightInputToHidden[k][g] = (n.learningRate * hiddenEror[k] * X[g]) + (oldWeightInputToHidden[k][g]* n.momemtum);
			 }
			 n.weightBiasHidden[k] += n.learningRate * hiddenEror[k];
		 	 }

		 
		 //calculating final weights
		 	for (int i=0; i<number_Outputs; i++)
		 	{
			 for (int k=0; k<number_Hidden; k++)
			 {
			 n.weightHiddenToOutput[k][i] = n.weightHiddenToOutput[k][i] + newWeightHiddenToOutput[k][i];
			 }
			 }
			 for (int k=0; k<number_Hidden; k++)
			 {
			 for (int i=0; i<number_Inputs; i++)
			 {
			 n.weightInputToHidden[i][k] = n.weightInputToHidden[i][k] + newWeightInputToHidden[k][i];
			 }
			 }
	 
 }
	 
	 System.out.println("epoch =" + epoch + "\t Error =" + myError);
	 		
 	}
 			
 			if (epoch==10000)
 			{
 			System.out.println("Error");
 			}
 			
 }
 			public static void main(String[] args) {
 			BackpropogationAlgorithm n1 = new BackpropogationAlgorithm() ;
 			n1.backPropagation();
 
}

}

