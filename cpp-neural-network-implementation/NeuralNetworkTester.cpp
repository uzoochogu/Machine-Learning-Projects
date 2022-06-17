#include <iostream>
#include "MLP.h"

int main() 
{
  srand(time(NULL));                                            
  rand();


  std::cout << "\n\n--------Logic Gate Example----------------\n\n";
  Perceptron *p = new Perceptron(2);

  //{10,10,-15} #AND
  //{15,15,-10}  #OR
  //{-15,-15,10}  #NOR
  //{-10,-10,15} #NAND

  p->set_weights({15,15,-10});

  std::cout << "Gate: "<<endl;
  std::cout<<p->run({0,0})<<endl;
  std::cout<<p->run({0,1})<<endl;
  std::cout<<p->run({1,0})<<endl;
  std::cout<<p->run({1,1})<<endl;

  std::cout<<"\n\n--------Hardcoded XOR Example----------------\n\n";
  MultiLayerPerceptron mlp = MultiLayerPerceptron({2,2,1});  //mlp
  mlp.set_weights({{{-10,-10,15},{15,15,-10}}, {{10,10,-15}}});
  std::cout << "Hard-coded weights:\n";
  mlp.print_weights();

  std::cout<<"XOR:"<<endl;
  std::cout<<"0 0 = "<<mlp.run({0,0})[0]<<endl;
  std::cout<<"0 1 = "<<mlp.run({0,1})[0]<<endl;
  std::cout<<"1 0 = "<<mlp.run({1,0})[0]<<endl;
  std::cout<<"1 1 = "<<mlp.run({1,1})[0]<<endl; 
}

