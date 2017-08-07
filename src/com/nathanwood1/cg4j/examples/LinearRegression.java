package com.nathanwood1.cg4j.examples;

import com.nathanwood1.cg4j.Eval;
import com.nathanwood1.cg4j.Tensor;
import com.nathanwood1.cg4j.nodes.Node;
import com.nathanwood1.cg4j.nodes.io.InputNode;
import com.nathanwood1.cg4j.nodes.io.VariableNode;
import com.nathanwood1.cg4j.nodes.math.AdditionNode;
import com.nathanwood1.cg4j.nodes.tensor.MeanNode;
import com.nathanwood1.cg4j.nodes.math.MultiplicationNode;
import com.nathanwood1.cg4j.nodes.math.SquareNode;
import com.nathanwood1.cg4j.nodes.math.SubtractionNode;
import com.nathanwood1.cg4j.optimizers.AdamOptimizer;
import com.nathanwood1.cg4j.optimizers.GradientDescentOptimizer;

import java.util.HashMap;

public class LinearRegression {

    public static void main(String[] args) {
        /*
         * Create the variables.
         * Variables can be tweaked to find optimal values.
         * Here, they represent 'y = mx + c'.
         * We are trying to find the optimal values for our data set.
         */
        VariableNode m = new VariableNode(new Tensor(new float[1], new int[]{1}));
        VariableNode c = new VariableNode(new Tensor(new float[1], new int[]{1}));

        /*
         * Create an InputNode.
         * This node is used to feed data into the graph.
         * Here, it represents the x in 'y = mx + c'.
         */
        InputNode x = new InputNode(new int[]{-1, 1});

        /*
         * This is where we program in the formula 'y = mx + c'
         */
        Node y = new MultiplicationNode(x, m);
        y = new AdditionNode(y, c);

        /*
         * 'yTarget' is the optimal value for 'y'.
         * We find the mean squared cost through the code below.
         */
        InputNode yTarget = new InputNode(new int[]{-1, 1});
        Node cost = new MeanNode(new SquareNode(new SubtractionNode(yTarget, y)));

        /*
         * Create a GradientDescentOptimizer and allow it to tweak 'm' and 'c'.
         * We also set the learning rate and what variable to minimize.
         */
        GradientDescentOptimizer optimizer = new GradientDescentOptimizer()
                .tweak(m)
                .tweak(c);
        optimizer.learningRate = 0.001f;

        /*
         * Create the deltas and use them to minimize 'cost'.
         */
        HashMap<VariableNode, Node> deltas = new HashMap<>();
        cost.createGradients(deltas, null);
        optimizer.minimize(cost, deltas);

        Tensor[] vals = {
                new Tensor(new float[]{
                        9,
                        8,
                        6,
                        14,
                        12,
                        10
                }, new int[]{6, 1}),
                new Tensor(new float[]{
                        4,
                        2,
                        1,
                        8,
                        1,
                        7
                }, new int[]{6, 1})
        };

        for (int i = 0; i < 4000; i++) {
            /*
             * Create an evaluator with input values
             */
            Eval eval = new Eval()
                    .addInput(x, vals[0])
                    .addInput(yTarget, vals[1]);

            /*
             * Allow the gradient descent algorithm to run.
             */
            optimizer.run(eval);

            /*
             * Display the cost every 100 iterations.
             */
            if (i % 100 == 0) {
                System.out.printf("Error: %f\n", eval.evaluate(cost).getVal(0));
            }
        }

        /*
         * Display the data.
         */
        Eval eval = new Eval()
                .addInput(x, vals[0])
                .addInput(yTarget, vals[1]);
        System.out.println();
        System.out.println("Cost: " + eval.evaluate(cost).getVal(0));
        System.out.println("y = " + eval.evaluate(m).getVal(0) + "x + " + eval.evaluate(c).getVal(0));
    }
}
