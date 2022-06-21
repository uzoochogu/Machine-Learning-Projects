#include "MLP.h"

double frand()
{
	return (2.0*(double)rand() / RAND_MAX) - 1.0;
}


// Return a new Perceptron object with the specified number of inputs (+1 for the bias).
Perceptron::Perceptron(int inputs, double bias)
{
	this->bias = bias;
	weights.resize(inputs+1);      
	generate(weights.begin(),weights.end(),frand);
}

// Run the perceptron. x is a vector with the input values.
double Perceptron::run(vector<double> x)
{
	x.push_back(bias);
	double sum = inner_product(x.begin(),x.end(),weights.begin(),(double)0.0);
	return sigmoid(sum);
}

// Set the weights. w_init is a vector with the weights from all neurons in previous layer.
void Perceptron::set_weights(vector<double> w_init)
{
	weights = w_init;
}

// Evaluate the sigmoid function for the floating point input x.
double Perceptron::sigmoid(double x)
{
	return 1.0/(1.0 + exp(-x));
}


// Return a new MultiLayerPerceptron object with the specified parameters.
MultiLayerPerceptron::MultiLayerPerceptron(vector<int> layers, double bias, double eta) 
{
    this->layers = layers;
    this->bias = bias;
    this->eta = eta;

    //Create the neurons layer by layer
    for (unsigned int i = 0; i < layers.size(); i++)
    {
        values.push_back(vector<double>(layers[i],0.0));                        //zero output for every neuron per layer
        deltas.push_back(vector<double>(layers[i],0.0));                        //zero errors on creation
        network.push_back(vector<Perceptron>());                                //initial empty perceptrons in network
      
        if (i > 0)                                                              //network[0] is the input layer,no neurons and perceptrons
            for (unsigned int j = 0; j < layers[i]; j++)                                 
                network[i].push_back(Perceptron(layers[i-1], bias));            //create perceptrons per neuron, add bias
    }
}

// Sets the weights. w_init is a vector of vectors of vectors with the weights for all but the input layer. 
void MultiLayerPerceptron::set_weights(vector<vector<vector<double> > > w_init) 
{
    for (int i = 0; i< w_init.size(); i++)
        for (unsigned int j = 0; j < w_init[i].size(); j++)
            network[i+1][j].set_weights(w_init[i][j]);
}


void MultiLayerPerceptron::print_weights() 
{
    cout << endl;
    for (int i = 1; i < network.size(); i++){
        for (unsigned int j = 0; j < layers[i]; j++) {
            cout << "Layer " << i+1 << " Neuron " << j << ": ";
            for (auto &it: network[i][j].weights)
                cout << it <<"   ";
            cout << endl;
        }
    }
    cout << endl;
}

//Feed a sample x into the Multilayer Perceptron.
vector<double> MultilayerPerceptron::run(vector<double> x)
{
  values[0] = x;                                         //input layer
  for(unsigned int i = 1; i < network.size(); i++)
  {
    values[i][j] = network[i][j].run(values[i-1])
  }
  return values.back()    
}


// Run a single (x,y) pair with the backpropagation algorithm.
double MultiLayerPerceptron::bp(vector<double> x, vector<double> y){
    
    // Backpropagation Step by Step:
    
    // STEP 1: Feed a sample to the network
    vector<double> outputs = run(x);
    
    // STEP 2: Calculate the MSE
    vector<double> error;
    double MSE = 0.0;
    for (int i = 0; i < y.size(); i++){
        error.push_back(y[i] - outputs[i]);
        MSE += error[i] * error[i];
    }
    MSE /= layers.back();

    // STEP 3: Calculate the output error terms
    for (int i = 0; i < outputs.size(); i++)
        delta.back()[i] = outputs[i] * (1 - outputs[i]) * (error[i]);

    // STEP 4: Calculate the error term of each unit on each layer    
    for (int i = network.size()-2; i > 0; i--)
        for (int h = 0; h < network[i].size(); h++){
            double fwd_error = 0.0;
            for (int k = 0; k < layers[i+1]; k++)
                fwd_error += network[i+1][k].weights[h] * delta[i+1][k];
            delta[i][h] = values[i][h] * (1-values[i][h]) * fwd_error;
        }
    
    // STEPS 5 & 6: Calculate the deltas and update the weights
    for (int i = 1; i < network.size(); i++)
        for (int j = 0; j < layers[i]; j++)
            for (int k = 0; k < layers[i-1]+1; k++){
                double cost;
                if (k==layers[i-1])
                    cost = eta * delta[i][j] * bias;
                else
                    cost = eta * delta[i][j] * values[i-1][k];
                network[i][j].weights[k] += cost;
            }
    return MSE;
}

