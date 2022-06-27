#include <iostream>
#include "MLP.hpp"

int main() 
{
  srand(static_cast<unsigned int>(time(NULL)));                                            
  rand();

  double MSE;


  //Test Code for a Segment Display Recognition System for a 7 Segment Display

  //Network Structure 1: 7,7,1
  int epochs = 1000;
  MultiLayerPerceptron *sdrnn7_to_1 = new MultiLayerPerceptron({7,7,1});

  // Dataset for the 7 to 1 network
  for (int i = 0; i < epochs; i++)
  {
      MSE = 0.0;
      MSE += sdrnn7_to_1->bp({1,1,1,1,1,1,0}, {0.05}); //0 pattern
      MSE += sdrnn7_to_1->bp({0,1,1,0,0,0,0}, {0.15}); //1 pattern
      MSE += sdrnn7_to_1->bp({1,1,0,1,1,0,1}, {0.25}); //2 pattern
      MSE += sdrnn7_to_1->bp({1,1,1,1,0,0,1}, {0.35}); //3 pattern
      MSE += sdrnn7_to_1->bp({0,1,1,0,0,1,1}, {0.45}); //4 pattern
      MSE += sdrnn7_to_1->bp({1,0,1,1,0,1,1}, {0.55}); //5 pattern
      MSE += sdrnn7_to_1->bp({1,0,1,1,1,1,1}, {0.65}); //6 pattern
      MSE += sdrnn7_to_1->bp({1,1,1,0,0,0,0}, {0.75}); //7 pattern
      MSE += sdrnn7_to_1->bp({1,1,1,1,1,1,1}, {0.85}); //8 pattern
      MSE += sdrnn7_to_1->bp({1,1,1,1,0,1,1}, {0.95}); //9 pattern
  }
  MSE /= 10.0;
  cout << endl << "7 to 1  network MSE: " << MSE << endl;

  

  //Network Structure 2: 7,7,10
  MultiLayerPerceptron *sdrnn7_to_10 = new MultiLayerPerceptron({7,7,10});

  //Dataset for the 7 to 10 network  
  for (int i = 0; i < epochs; i++)
  {
      MSE = 0.0;
      MSE += sdrnn7_to_10->bp({1,1,1,1,1,1,0}, {1,0,0,0,0,0,0,0,0,0}); //0 pattern
      MSE += sdrnn7_to_10->bp({0,1,1,0,0,0,0}, {0,1,0,0,0,0,0,0,0,0}); //1 pattern
      MSE += sdrnn7_to_10->bp({1,1,0,1,1,0,1}, {0,0,1,0,0,0,0,0,0,0}); //2 pattern
      MSE += sdrnn7_to_10->bp({1,1,1,1,0,0,1}, {0,0,0,1,0,0,0,0,0,0}); //3 pattern
      MSE += sdrnn7_to_10->bp({0,1,1,0,0,1,1}, {0,0,0,0,1,0,0,0,0,0}); //4 pattern
      MSE += sdrnn7_to_10->bp({1,0,1,1,0,1,1}, {0,0,0,0,0,1,0,0,0,0}); //5 pattern
      MSE += sdrnn7_to_10->bp({1,0,1,1,1,1,1}, {0,0,0,0,0,0,1,0,0,0}); //6 pattern
      MSE += sdrnn7_to_10->bp({1,1,1,0,0,0,0}, {0,0,0,0,0,0,0,1,0,0}); //7 pattern
      MSE += sdrnn7_to_10->bp({1,1,1,1,1,1,1}, {0,0,0,0,0,0,0,0,1,0}); //8 pattern
      MSE += sdrnn7_to_10->bp({1,1,1,1,0,1,1}, {0,0,0,0,0,0,0,0,0,1}); //9 pattern
  }
  MSE /= 10.0;
  cout << "7 to 10 network MSE: " << MSE << endl;

    
  // Dataset for the 7 to 7 network
  MultiLayerPerceptron *sdrnn7_to_7 = new MultiLayerPerceptron({7,7,7});

  for (int i = 0; i < epochs; i++)
  {
      MSE = 0.0;
      MSE += sdrnn7_to_7->bp({1,1,1,1,1,1,0}, {1,1,1,1,1,1,0}); //0 pattern
      MSE += sdrnn7_to_7->bp({0,1,1,0,0,0,0}, {0,1,1,0,0,0,0}); //1 pattern
      MSE += sdrnn7_to_7->bp({1,1,0,1,1,0,1}, {1,1,0,1,1,0,1}); //2 pattern
      MSE += sdrnn7_to_7->bp({1,1,1,1,0,0,1}, {1,1,1,1,0,0,1}); //3 pattern
      MSE += sdrnn7_to_7->bp({0,1,1,0,0,1,1}, {0,1,1,0,0,1,1}); //4 pattern
      MSE += sdrnn7_to_7->bp({1,0,1,1,0,1,1}, {1,0,1,1,0,1,1}); //5 pattern
      MSE += sdrnn7_to_7->bp({1,0,1,1,1,1,1}, {1,0,1,1,1,1,1}); //6 pattern
      MSE += sdrnn7_to_7->bp({1,1,1,0,0,0,0}, {1,1,1,0,0,0,0}); //7 pattern
      MSE += sdrnn7_to_7->bp({1,1,1,1,1,1,1}, {1,1,1,1,1,1,1}); //8 pattern
      MSE += sdrnn7_to_7->bp({1,1,1,1,0,1,1}, {1,1,1,1,0,1,1}); //9 pattern
  }
  MSE /= 10.0;
  cout << "7 to 7  network MSE: " << MSE << endl << endl;


  
  //using the networks
  int option = 0;
  double a,b,c,d,e,f,g ;
  std::vector<double> result;
  do
  {
    std::cout << "\n\nTest the Segment Display Recognition System\n"; 
    std::cout << "1. Use the Seven to One Neural Net\n";
    std::cout << "2. Use the Seven to Ten Neural Net\n";
    std::cout << "3. Use the Seven to Seven Neural Net\n";
    std::cout << "4. Exit\n";
    std::cout << "Choose Model:\nChoice: ";
    std::cin >> option;

    if(option >= 1 && option <= 3)    //only select choice between 1 and 3
    {
      std::cout << "\nInput the values (0-1) for the seven segments from a-g(warning: no validation): ";
      std::cin >> a >> b >> c >> d >> e >> f >> g;
    }

    switch(option)
    {
      case 1:
        result = sdrnn7_to_1->run({a,b,c,d,e,f,g});
        std::cout << "\nPrediction = " << result[0] << std::endl;
        break;

      case 2:
        result = sdrnn7_to_10->run({a,b,c,d,e,f,g});
        std::cout << "Prediction = " 
                  << std::distance(result.begin(), 
                                   std::max_element(result.begin(), result.end()));    //returns position of max element.
        std::cout << "\nAll the values are:\n";
        for(auto i:result )
        {
          std::cout << i << "\n";
        }
        std::cout << std::endl;
        break;

      case 3:
        result = sdrnn7_to_7->run({a,b,c,d,e,f,g});
        std::cout << "Compare the values:\n";
        for(auto i:result )
        {
          std::cout << i << "\n";
        }
        std::cout << std::endl;
        break;

      case 4:
        break;

      default:
        std::cout << "Wrong Choice! (Please enter a number between 1-3)";
        option = 4;
    }
    
  }
  while(option != 4);
  delete(sdrnn7_to_10);
  delete(sdrnn7_to_1);
  delete(sdrnn7_to_7);
}