import java.util.Random;

public class Neuron {
public int number_Inputs=2;
public int number_Hidden=4;
public int number_Outputs=1;

public double momemtum=0.0;
public double learningRate=0.2;

double[] outputFromIn= new double[number_Inputs];//Output from inputs which is given as input to hidden layer
double outputError;

double finalError=0.05;
int epoch=0;

public double [][]weightInputToHidden=new double[number_Inputs][number_Hidden];    // weights from the input to hidden layer
public double [][]weightHiddenToOutput=new double[number_Hidden][number_Outputs];   //weights from hidden to output layer

public double[] weightBiasHidden= new double[number_Hidden]; // bias given to the hidden layer
public double[] weightBiasOutput= new double[number_Outputs];// bias given to the output layer

public void calculateRandomWeight()
{    //Random weight update from input to hidden including bias
	
	for (int i=0; i<number_Hidden; i++) 
	{
		for (int j=0; j<number_Inputs; j++)
		{
			Random r =new Random();
			weightInputToHidden[j][i]= r.nextDouble() - 0.5;  // assigning random weights to the variables
			
		}
		Random rand =new Random();
		weightBiasHidden[i] = rand.nextDouble() -0.5;
	}
	
	
	
	
		for (int k=0; k<number_Outputs; k++)   //weight update from hidden to output including bias

		{
			for (int j=0; j<number_Hidden; j++) 
			{
			
			Random r =new Random();
			weightHiddenToOutput[j][k]= r.nextDouble()-0.5;
			
			}
			Random rand =new Random();
			weightBiasOutput[k] = rand.nextDouble() -0.5;
	    }

}





}
