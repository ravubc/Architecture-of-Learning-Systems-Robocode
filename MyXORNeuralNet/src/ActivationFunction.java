
public class ActivationFunction
{
	
public double bipolarSigmoid(double x)    // sigmoid activation function for bipolar inputs
	
	{
		return ((2.0/(1+Math.exp(-x)))-1);
   }
public double binarySigmoid(double x)     // sigmoid activation function for binary inputs
	
	{
		return (1.0/(1+(Math.exp(-x))));
   }
	
double derivativeBinarySigmoid(double x)    // derivative of the binary sigmoid
{
	return (binarySigmoid(x)*(1-binarySigmoid(x)));
	
}
double derivativeBipolarSigmoid(double x)  //derivative of the bipolar sigmoid
{
	return ( (1+bipolarSigmoid(x)) * 0.5 * (1-bipolarSigmoid(x)) );
	
}

}

