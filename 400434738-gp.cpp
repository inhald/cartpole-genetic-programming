#include <bits/stdc++.h>
#include <math.h>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stack>
#include <vector>

#include "cartCentering.h"

using namespace std;

// return a double unifomrly sampled in (0,1)
double randDouble(mt19937& rng) {
  return std::uniform_real_distribution<>{0, 1}(rng);
}
// return uniformly sampled 0 or 1
bool randChoice(mt19937& rng) {
  return std::uniform_int_distribution<>{0, 1}(rng);
}
// return a random integer uniformly sampled in (min, max)
int randInt(mt19937& rng, const int& min, const int& max) {
  return std::uniform_int_distribution<>{min, max}(rng);
}

// return true if op is a suported operation, otherwise return false
bool isOp(string op) {
  if (op == "+")
    return true;
  else if (op == "-")
    return true;
  else if (op == "*")
    return true;
  else if (op == "/")
    return true;
  else if (op == ">")
    return true;
  else if (op == "abs")
    return true;
  else
    return false;
}

int arity(string op) {
  if (op == "abs")
    return 1;
  else
    return 2;
}

void externalContent(stringstream& ss, mt19937& rng){
   bool numorvar = randChoice(rng);
    //picking num or var
    if(numorvar){
      bool ab = randChoice(rng);
      //pick a or b
      if(!ab){
         ss << "a ";
      }
      else {
        ss << "b ";
      }
    }
    //passing randDouble to stringstream
   else{
        double number = randDouble(rng);
        ss << number << " ";
    }
}


typedef string Elem;

class LinkedBinaryTree {
 public:
  struct Node {
    Elem elt;
    string name;
    Node* par;
    Node* left;
    Node* right;
    Node() : elt(), par(NULL), name(""), left(NULL), right(NULL) {}
    int depth() {
      if (par == NULL) return 0;
      return par->depth() + 1;
    }
  };

  class Position {
   private:
    Node* v;

   public:
    Position(Node* _v = NULL) : v(_v) {}
    Elem& operator*() { return v->elt; }
    Position left() const { return Position(v->left); }
    void setLeft(Node* n) { v->left = n; }
    Position right() const { return Position(v->right); }
    void setRight(Node* n) { v->right = n; }

    Node* getNode(){return v;}

    

    Position parent() const  // get parent
    {
      return Position(v->par);
    }
    bool isRoot() const  // root of the tree?
    {
      return v->par == NULL;
    }
    bool isExternal() const  // an external node?
    {
      return v->left == NULL && v->right == NULL;
    }
    friend class LinkedBinaryTree;  // give tree access
  };
  typedef vector<Position> PositionList;

 public:
  LinkedBinaryTree() : _root(NULL), score(0), steps(0), generation(0) {}

  // copy constructor
  LinkedBinaryTree(const LinkedBinaryTree& t) {
    _root = copyPreOrder(t.root());
    score = t.getScore();
    steps = t.getSteps();
    generation = t.getGeneration();
  }

  // copy assignment operator
  LinkedBinaryTree& operator=(const LinkedBinaryTree& t) {
    if (this != &t) {
      // if tree already contains data, delete it
      if (_root != NULL) {
        PositionList pl = positions();
        for (auto& p : pl) delete p.v;
      }
      _root = copyPreOrder(t.root());
      score = t.getScore();
      steps = t.getSteps();
      generation = t.getGeneration();
    }
    return *this;
  }

  // destructor
  ~LinkedBinaryTree() {
    if (_root != NULL) {
      PositionList pl = positions();
      for (auto& p : pl) delete p.v;
    }
  }

  int size() const { return size(_root); }
  int size(Node* root) const;
  int depth() const;
  bool empty() const { return size() == 0; };
  Node* root() const { return _root; }
  PositionList positions() const;
  void addRoot() { _root = new Node; }
  void addRoot(Elem e) {
    _root = new Node;
    _root->elt = e;
  }
  void nameRoot(string name) { _root->name = name; }
  void addLeftChild(const Position& p, const Node* n);
  void addLeftChild(const Position& p);
  void addRightChild(const Position& p, const Node* n);
  void addRightChild(const Position& p);
  void printExpression() { printExpression(_root); }
  void printExpression(Node* v);
  double evaluateExpression(double a, double b) {
    return evaluateExpression(Position(_root), a, b);
  };
  double evaluateExpression(const Position& p, double a, double b);
  long getGeneration() const { return generation; }
  void setGeneration(int g) { generation = g; }
  double getScore() const { return score; }
  void setScore(double s) { score = s; }
  double getSteps() const { return steps; }
  void setSteps(double s) { steps = s; }
  void randomExpressionTree(Node* p, const int& maxDepth, mt19937& rng);
  void randomExpressionTree(const int& maxDepth, mt19937& rng) {
    randomExpressionTree(_root, maxDepth, rng);
  }
  void deleteSubtreeMutator(mt19937& rng);
  void addSubtreeMutator(mt19937& rng, const int maxDepth);
  //added functions
  void deleteSubtree(Node* subtreeRoot);

  vector<LinkedBinaryTree> crossover(const LinkedBinaryTree& parent1, const LinkedBinaryTree& parent2, mt19937& rng);


 protected:                                        // local utilities
  void preorder(Node* v, PositionList& pl) const;  // preorder utility
  Node* copyPreOrder(const Node* root);
  double score;     // mean reward over 20 episodes
  double steps;     // mean steps-per-episode over 20 episodes
  long generation;  // which generation was tree "born"
 private:
  Node* _root;  // pointer to the root
};



vector<LinkedBinaryTree> LinkedBinaryTree::crossover(const LinkedBinaryTree& parent1, const LinkedBinaryTree& parent2, mt19937& rng){
  /*To do this, I will select two parents with more than 1 node and swap two 
  subtrees at non-root nodes. I will then append the resulting trees into my vector.*/
  vector<LinkedBinaryTree> children; 

  LinkedBinaryTree child1 = parent1;
  LinkedBinaryTree child2 = parent2; 
  
  //finding subtree root nodes. we want these to be operators
  int index1 = randInt(rng, 0, child1.size()-1);
  PositionList nodes1 = child1.positions();
  Position deletePos1=nodes1[index1];
  Node* subtreeRoot1=deletePos1.v;

  if(subtreeRoot1 == child1.root()) subtreeRoot1 = subtreeRoot1->left;
  //it will be ensured that trees have a depth > 0

  int index2= randInt(rng, 0, child2.size()-1);
  PositionList nodes2 = child2.positions();
  Position deletePos2 = nodes2[index2];
  Node* subtreeRoot2= deletePos2.v;

  if(subtreeRoot2 == child2.root()) subtreeRoot2 = subtreeRoot2->right;


  //swapping subtrees
  subtreeRoot1 = copyPreOrder(subtreeRoot2);

  subtreeRoot2 = copyPreOrder(subtreeRoot1);

  deleteSubtree(subtreeRoot1);
  deleteSubtree(subtreeRoot2);

  children.push_back(child1);
  children.push_back(child2);

  return children;

}


//Comparator ADT

class LexLessThan {
  public:
    bool operator()(LinkedBinaryTree Ta, LinkedBinaryTree Tb){
      //taking abs of difference of scores
      if(abs(Ta.getScore() - Tb.getScore()) < 0.01){

        int tree_size_a = Ta.size();
        int tree_size_b = Tb.size();

        //comparing sizes
        if(tree_size_a > tree_size_b){
          return true;
        }
        else{
          return false;
        }
      }
      else{
        //in this case, I just compare scores
        if(Tb.getScore() > Ta.getScore()){
          return true;
        }
        else{
          return false;
        }
      }

    }

};

LinkedBinaryTree createRandExpressionTree(int max_depth, mt19937& rng);

// add the tree rooted at node child as this tree's left child
void LinkedBinaryTree::addLeftChild(const Position& p, const Node* child) {
  Node* v = p.v;
  v->left = copyPreOrder(child);  // deep copy child
  v->left->par = v;
}

// add the tree rooted at node child as this tree's right child
void LinkedBinaryTree::addRightChild(const Position& p, const Node* child) {
  Node* v = p.v;
  v->right = copyPreOrder(child);  // deep copy child
  v->right->par = v;
}

void LinkedBinaryTree::addLeftChild(const Position& p) {
  Node* v = p.v;
  v->left = new Node;
  v->left->par = v;
}

void LinkedBinaryTree::addRightChild(const Position& p) {
  Node* v = p.v;
  v->right = new Node;
  v->right->par = v;
}

// return a list of all nodes
LinkedBinaryTree::PositionList LinkedBinaryTree::positions() const {
  PositionList pl;
  preorder(_root, pl);
  return PositionList(pl);
}

void LinkedBinaryTree::preorder(Node* v, PositionList& pl) const {
  pl.push_back(Position(v));
  if (v->left != NULL) preorder(v->left, pl);
  if (v->right != NULL) preorder(v->right, pl);
}

int LinkedBinaryTree::size(Node* v) const {
  int lsize = 0;
  int rsize = 0;
  if (v->left != NULL) lsize = size(v->left);
  if (v->right != NULL) rsize = size(v->right);
  return 1 + lsize + rsize;
}

int LinkedBinaryTree::depth() const {
  PositionList pl = positions();
  int depth = 0;
  for (auto& p : pl) depth = std::max(depth, p.v->depth());
  return depth;
}

LinkedBinaryTree::Node* LinkedBinaryTree::copyPreOrder(const Node* root) {
  if (root == NULL) return NULL;
  Node* nn = new Node;
  nn->elt = root->elt;
  nn->left = copyPreOrder(root->left);
  if (nn->left != NULL) nn->left->par = nn;
  nn->right = copyPreOrder(root->right);
  if (nn->right != NULL) nn->right->par = nn;
  return nn;
}

void LinkedBinaryTree::printExpression(Node* v) {
  // replace print statement with your code

  //v won't be null but this is just a safety thing
  if(v != NULL){
    //if external just print it
    if(v->left == NULL && v->right == NULL){
      cout << v->elt;
    }
    else{
      //handling case where it is a binary operation
      if(isOp(v->elt) && (arity(v->elt) > 1)){ //means it is a non-abs
        cout << "(";
        printExpression(v->left);
        cout << " " <<  v->elt << " ";
        printExpression(v->right);
        cout << ")";
      }
      //case where it is abs
    if(isOp(v->elt) && (arity(v->elt) == 1)){
        cout << v->elt;
        cout << "(";
        printExpression(v->left);
        cout << ")";
      }
    }
  }
}

double evalOp(string op, double x, double y = 0) {
  double result;
  if (op == "+")
    result = x + y;
  else if (op == "-")
    result = x - y;
  else if (op == "*")
    result = x * y;
  else if (op == "/") {
    result = x / y;
  } else if (op == ">") {
    result = x > y ? 1 : -1;
  } else if (op == "abs") {
    result = abs(x);
  } else
    result = 0;
  return isnan(result) || !isfinite(result) ? 0 : result;
}

double LinkedBinaryTree::evaluateExpression(const Position& p, double a,
                                            double b) {
  if (!p.isExternal()) {
    auto x = evaluateExpression(p.left(), a, b);
    if (arity(p.v->elt) > 1) {
      auto y = evaluateExpression(p.right(), a, b);
      return evalOp(p.v->elt, x, y);
    } else {
      return evalOp(p.v->elt, x);
    }
  } else {
    if (p.v->elt == "a")
      return a;
    else if (p.v->elt == "b")
      return b;
    else
      return stod(p.v->elt);
  }
}


void LinkedBinaryTree::deleteSubtree(Node* subtreeRoot){

  if(subtreeRoot != NULL){
    if(subtreeRoot->left != NULL){
      deleteSubtree(subtreeRoot->left);
    }
    if(subtreeRoot->right != NULL){
      deleteSubtree(subtreeRoot->right);
    }

    delete subtreeRoot;
  }

}


void LinkedBinaryTree::deleteSubtreeMutator(mt19937& rng) {
  
  /*it is easier to select an internal node and one of its children
  as selecting an external node will require remove its parent
  anyway.*/

  /*in addition this makes it easier to handle the case where
  a single node treeis passed through the function*/


  //determining root node randomly
  PositionList nodes = positions();
  int index = randInt(rng, 0, nodes.size()-1);
  Position deletePos = nodes[index];
  Node* selected = deletePos.v;
  bool left;

  /*if a non-operator is chosen, then shift the selection to its parent.
  I must ensure that the tree has more than one node or I seg fault.*/
  if(!isOp(selected->elt) && nodes.size() > 1){
    if(selected->par != NULL) selected=selected->par;
  }


  if(isOp(selected->elt)){
     //delete right child
    //there will be two cases, arity = 1 or arity = 2
    if(arity(selected->elt) > 1){
      Node* rchild = selected->right;
      Node* lchild = selected->left;
      Node* parent = selected->par;

      //if root is chosen, then shift selection to left child
      //deleteSubtree cleans up unneeded memory
      if(selected == _root){
        _root = copyPreOrder(lchild);
        _root->par = NULL;
        deleteSubtree(rchild);
      }
      else{
        //must know if child is on left or right
        if(parent->left == selected){
          left = true;
        }
        else{
          left = false;
        }

        //if right then right grandchild is attached to parent node
        if(!left){
          parent->right = copyPreOrder(rchild);
          (parent->right)->par = parent;
          deleteSubtree(lchild);
        }
        else{
          //if left then left grandchild is attached to parent node
          parent->left = copyPreOrder(lchild);
          (parent->left)->par = parent;
          deleteSubtree(rchild);

        }
      }
    }
    else{
      //if arity = 1 and root is chosen, then move to left child
      Node* lchild = selected->left;
      if(selected == _root){
        _root = copyPreOrder(lchild);
        _root->par = NULL;
      }
      else{
        //if not then correct parent to left child
          Node* parent = selected->par;
          parent->left = copyPreOrder(lchild);
          parent->left->par = parent;
      }
      

    }
  }
  selected = NULL;



}


void LinkedBinaryTree::addSubtreeMutator(mt19937& rng, const int maxDepth) {

  /*to do this, I will simply replace a random external node with a randomly 
  generated binary operation tree and ensure that result is less than max depth*/

  //generating
  LinkedBinaryTree addedTree = createRandExpressionTree(maxDepth, rng);
  int added_depth = addedTree.depth();

  //we use the max depth of the addedTree to ensure
  //that the new Tree will be less than max depth
  while(added_depth == 0 || depth() + added_depth > maxDepth){
    addedTree = createRandExpressionTree(maxDepth, rng);
    added_depth = addedTree.depth();
  }

  //there will be two cases, case I: if binary operation tree has depth > 0
  //and case II: binary operation tree has single node.

  //case I:
  if(depth() > 0){

    //I will move in random direction from root
    //until I reach external node
    Node* cur = _root;
    bool choice;
    bool left;
    
    //finding external node
    while(!(cur->right == NULL && cur->left == NULL)){
      choice = randChoice(rng);
      if(!choice){
        //safety
        if(cur->left == NULL) continue;
        else {
          cur=cur->left;
          bool left = true;
        }
      }
      else{
        //safety
        if(cur->right == NULL) continue;
        else {
          cur=cur->right;
          bool left = false;
        }
      }
    }

    //Now I will append the random tree 
    //to ensure tree is valid, this node must be replace with tree
    Node* parent = cur->par;
    //Case I: arity > 1
    if(arity(parent->elt) > 1){
      //left gives the direction of the last travelled direction
      if(left){
        delete parent->left;  
        parent->left = copyPreOrder(addedTree.root());
        parent->left->par = parent;
      }
      else{
        delete parent->right;
        parent->right = copyPreOrder(addedTree.root());
        parent->right->par = parent;
      }
    }
    //if arity = 1, then it will only have left child
    else{
      delete parent->left;
      parent->left = copyPreOrder(addedTree.root());
      parent->left->par = parent;
    }


   
  }
  else{
    //if it is a single node tree, it is better to just replace it entirely
    _root = copyPreOrder(addedTree.root());
    _root->par = NULL;

  }


}

bool operator<(const LinkedBinaryTree& x, const LinkedBinaryTree& y) {
  return x.getScore() < y.getScore();
}

LinkedBinaryTree createExpressionTree(string postfix) {
  stack<LinkedBinaryTree> tree_stack;
  stringstream ss(postfix);
  // Split each line into words
  string token;
  while (getline(ss, token, ' ')) {
    LinkedBinaryTree t;
    if (!isOp(token)) {
      t.addRoot(token);
      tree_stack.push(t);
    } else {
      t.addRoot(token);
      if (arity(token) > 1) {
        LinkedBinaryTree r = tree_stack.top();
        tree_stack.pop();
        t.addRightChild(t.root(), r.root());
      }
      LinkedBinaryTree l = tree_stack.top();
      tree_stack.pop();
      t.addLeftChild(t.root(), l.root());
      tree_stack.push(t);
    }
  }
  return tree_stack.top();
}

LinkedBinaryTree createRandExpressionTree(int max_depth, mt19937& rng) {

  //stringstream will be used to hold the postfix expression
  string arityTwoOperations[] = {"+ ", "- ", "* ", "/ ", "> "};
  stringstream ss; 
  ss.clear();

  //init depth
  int depth = randInt(rng, 0, max_depth);
 


  //if depth is zero this is a single node tree which has a constant value
  /*externalContent function simply randomly generated either a variable (a or b)
  or a double*/ 
  if(depth == 0){
    externalContent(ss, rng);
  }
  else if(depth == 1){
    //there are two structures: abs(elem) or a binary operation tree
    bool choice = randChoice(rng);
    //case 1:
    if(!choice){
      externalContent(ss, rng);
      ss << "abs ";
    }
    //case 2:
    else{
      for(int i =0; i < 2; ++i){
        externalContent(ss, rng);
      }
      //I pick a random operation for the binary operation tree
      ss << arityTwoOperations[randInt(rng,0,4)];

    }
  }
  else{
    //if the depth is greater than two. this is more complicated


    /*the idea is to pick one of the initial structures with depth 0 or 1
    and randomly append it to either a binary operation tree or take the 
    //absolute value of it. This will increase the total_depth by either 1 or 2. 
    I want total_depth to eventually equal to depth*/

    //variable to pick initial structure
    bool initial = randChoice(rng);
    int total_depth;

    if(!initial){
      //a single node tree will have a depth of zero
      externalContent(ss, rng);
      total_depth = 0;
    }
    else{
      //a binary operation tree can be built similar to above and will
      //have a depth of 1.
      bool choice = randChoice(rng);

      if(!choice){
        externalContent(ss, rng);
        ss << "abs ";
      }

      else{
        for(int i =0; i < 2; ++i){
          externalContent(ss, rng);
        } 
        ss << arityTwoOperations[randInt(rng,0,4)];

      }
      total_depth = 1;
    }

    while(total_depth != depth){
      //it must be ensured that the operation is valid
      //in the first case, it will always be valid
      //in the second it might not. Thus I check if total_depth + 2 <= depth
      int addChoice = randInt(rng,0,5);

      if(addChoice == 5){
        ss << "abs ";
        total_depth += 1;
      }
      else if(total_depth + 2 <= depth){
        externalContent(ss, rng);
        ss << arityTwoOperations[addChoice];
        total_depth +=2;
      }
      else{
        continue;
      }
    }

  }

  //stringstream is converted to string and expression tree is created
  string str = ss.str();
  LinkedBinaryTree t = createExpressionTree(str);
  return t;
}

// evaluate tree t in the cart centering task
void evaluate(mt19937& rng, LinkedBinaryTree& t, const int& num_episode,
              bool animate) {
  cartCentering env;
  double mean_score = 0.0;
  double mean_steps = 0.0;
  for (int i = 0; i < num_episode; i++) {
    double episode_score = 0.0;
    int episode_steps = 0;
    env.reset(rng);
    while (!env.terminal()) {
      int action = t.evaluateExpression(env.getCartXPos(), env.getCartXVel());
      episode_score += env.update(action, animate);
      episode_steps++;
    }
    mean_score += episode_score;
    mean_steps += episode_steps;
  }
  t.setScore(mean_score / num_episode);
  t.setSteps(mean_steps / num_episode);
}

int main() {
  mt19937 rng(42);
  // Experiment parameters
  const int NUM_TREE = 50;
  const int MAX_DEPTH_INITIAL = 1;
  const int MAX_DEPTH = 20;
  const int NUM_EPISODE = 20;
  const int MAX_GENERATIONS = 100;

  // Create an initial "population" of expression trees
  vector<LinkedBinaryTree> trees;
  for (int i = 0; i < NUM_TREE; i++) {
    LinkedBinaryTree t = createRandExpressionTree(MAX_DEPTH_INITIAL, rng);
    trees.push_back(t);
  }

  // for(auto&t : trees){
  //   t.printExpression();
  //   cout << endl;
  // }

  // Genetic Algorithm loop
  LinkedBinaryTree best_tree;
  std::cout << "generation,fitness,steps,size,depth" << std::endl;
  for (int g = 1; g <= MAX_GENERATIONS; g++) {

    // Fitness evaluation
    for (auto& t : trees) {
      if (t.getGeneration() < g - 1) continue;  // skip if not new
      evaluate(rng, t, NUM_EPISODE, false);
    }

    // sort trees using overloaded "<" op (worst->best)
    std::sort(trees.begin(), trees.end());

    // // sort trees using comparaor class (worst->best)
    // std::sort(trees.begin(), trees.end(), LexLessThan());

    // erase worst 50% of trees (first half of vector)
    trees.erase(trees.begin(), trees.begin() + NUM_TREE / 2);

    // Print stats for best tree
    best_tree = trees[trees.size() - 1];
    std::cout << g << " ";
    std::cout << best_tree.getScore();
    std::cout << best_tree.getSteps() << ",";
    std::cout << best_tree.size() << ", ";
    std::cout << best_tree.depth() << std::endl;
    // cout << endl;

    LinkedBinaryTree parent1, parent2;
    bool canCross = false;
    //I must know if I can perform the crossover operation
    for(int i =0; i < trees.size()-1; ++i){
      int depth1 = trees[i].depth();

      for(int j=i+1; j < trees.size(); ++j){
        int depth2 = trees[j].depth();

        //I must have two distinct trees with greater than zero depth
        if(depth1 > 0 && depth2 > 0){
          canCross = true;
          break;
        }

      }
    }

    int selectionsize; 
    if(canCross) {selectionsize = NUM_TREE-2;}
    else {selectionsize = NUM_TREE;}

    // Selection and mutation
    while (trees.size() < selectionsize) {
      // Selected random "parent" tree from survivors
      LinkedBinaryTree parent = trees[randInt(rng, 0, (NUM_TREE / 2) - 1)];
      
      // Create child tree with copy constructor
      LinkedBinaryTree child(parent);
      child.setGeneration(g);
      
      // Mutation
      // Delete a randomly selected part of the child's tree
      child.deleteSubtreeMutator(rng);
      // Add a random subtree to the child
      child.addSubtreeMutator(rng, MAX_DEPTH);
      // child.printExpression();
      // cout << endl;
      
      trees.push_back(child);
    }

      //selecting parent trees for crossover
    if(canCross){
      
      //I require two randomly generated parent trees
      parent1 = trees[randInt(rng, 0, (NUM_TREE/2)-1)];
      parent2 = trees[randInt(rng, 0, (NUM_TREE/2)-1)];

      //These parents must both have non-zero depth
      while(parent1.depth() == 0 || parent2.depth() == 0){
        parent1 = trees[randInt(rng, 0, (NUM_TREE/2)-1)];
        parent2 = trees[randInt(rng, 0, (NUM_TREE/2)-1)];

      }

      LinkedBinaryTree dummy; 

      //children vector + calling crossover
      vector<LinkedBinaryTree> children = dummy.crossover(parent1, parent2, rng);

      //trees are appended into main vector
      for(int i=0; i < children.size(); ++i){
        trees.push_back(children[i]);
      }

    }




  }

  // // Evaluate best tree with animation
  const int num_episode = 3;
  evaluate(rng, best_tree, num_episode, true);

  // Print best tree info
  std::cout << std::endl << "Best tree:" << std::endl;
  best_tree.printExpression();
  std::cout << endl;
  std::cout << "Generation: " << best_tree.getGeneration() << endl;
  std::cout << "Size: " << best_tree.size() << std::endl;
  std::cout << "Depth: " << best_tree.depth() << std::endl;
  std::cout << "Fitness: " << best_tree.getScore() << std::endl << std::endl;
}
