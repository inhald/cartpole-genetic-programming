#include <bits/stdc++.h>

using namespace std;
#include <random>



int main(){

  vector<int> vect;
  int size = 30;

  for(int i =0; i < 30; ++i){
    vect.push_back(i);
  }

  vect.erase(vect.begin(), vect.begin() + size/2);


  for(int i =0; i < vect.size(); ++i){
    cout << vect[i] << endl;
  }


}




