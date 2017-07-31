# comp-graph-4J
Computational graph for Java.

***Please note that this is still WIP! Currently there are not enough nodes and no gradient decent!***

## What is a computational graph?

Normally, in computing, we use basic operations to calculate values. A computational graph will store the operations needed to operate and run them when it's needed.

Let's look at a simple example: adding numbers.

Without a computational graph, it would look like this:
```java
float c = a + b;
```
With a computational graph, it would look like this:
```java
InputNode a = new InputNode(new int[]{-1, 1});
InputNode b = new InputNode(new int[]{-1, 1});
Node c = new AdditionNode(a, b);
```
Note that no operation has taken place here. All we've done is specified what we want to do. To run it, we need to first specify inputs:
```java
Tensor aInput = new TensorInput(new float[]{
  1,
  2,
  3
}, new int[]{3, 1});
Tensor bInput = new TensorInput(new float[]{
  4,
  2,
  1
}, new int[]{3, 1});
```
Finally, we create an `Eval` to feed the data into the graph
```java
Eval eval = new Eval()
    .addInput(a, aInput)
    .addInput(b, bInput);

System.out.println(eval.evaluate(c).arrayToString());
```
This will return `[[5.0], [4.0], [4.0]]`. This was all ran by the computational graph.
## Why is this better‽

This may seem a little unnecessary and convoluted, but it is actually extremely helpful.

Because we first specified what we are doing, we can calculate the gradients and other useful things. This is mostly used for Neural Networks as (most of them) use gradient decent, reliant on calculus.

We can also 'batch operate' data. In the example above, we ran 3 sums simultaneously.

## Use

The main class you will use is the `Node` class. It represents a node in the computational graph.

## JavaDoc
The javadoc can be found at [https://nathanwood1.github.io/comp-graph-4J/javadoc/](https://nathanwood1.github.io/comp-graph-4J/javadoc/)
