package com.nathanwood1.cg4j.examples;

import com.nathanwood1.cg4j.Eval;
import com.nathanwood1.cg4j.Tensor;
import com.nathanwood1.cg4j.nodes.Node;
import com.nathanwood1.cg4j.nodes.io.InputNode;
import com.nathanwood1.cg4j.nodes.io.VariableNode;
import com.nathanwood1.cg4j.nodes.math.AdditionNode;
import com.nathanwood1.cg4j.nodes.math.MatrixMultiplicationNode;
import com.nathanwood1.cg4j.nodes.math.SquareNode;
import com.nathanwood1.cg4j.nodes.math.SubtractionNode;
import com.nathanwood1.cg4j.nodes.nn.SigmoidNode;
import com.nathanwood1.cg4j.nodes.tensor.MeanNode;
import com.nathanwood1.cg4j.optimizers.AdamOptimizer;
import com.nathanwood1.cg4j.optimizers.GradientDescentOptimizer;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Random;

public class XOR {

    public static void main(String[] args) {
        /*
         * Seed the randomness so it runs the same every time
         */
        Random seed = new Random(0);

        /*
         * Create the weights and biases
         */
        VariableNode layer1Weights = new VariableNode(Tensor.fromRandom(seed, -1, 1, new int[]{2, 4}));
        VariableNode layer1Biases = new VariableNode(Tensor.fromRandom(seed, -1, 1, new int[]{4}));
        VariableNode layer2Weights = new VariableNode(Tensor.fromRandom(seed, -1, 1, new int[]{4, 1}));
        VariableNode layer2Biases = new VariableNode(Tensor.fromRandom(seed, -1, 1, new int[]{1}));

        /*
         * Allows input of data into the graph.
         */
        InputNode x = new InputNode(new int[]{-1, 2});

        /*
         * This is the base for all fully-connected layers.
         */
        Node y = new MatrixMultiplicationNode(x, layer1Weights);
        y = new AdditionNode(y, layer1Biases);
        y = new SigmoidNode(y);

        y = new MatrixMultiplicationNode(y, layer2Weights);
        y = new AdditionNode(y, layer2Biases);
        y = new SigmoidNode(y);

        /*
         * Create a target and mean squared error.
         */
        InputNode yTarget = new InputNode(new int[]{-1, 1});
        Node cost = new MeanNode(new SquareNode(new SubtractionNode(yTarget, y)));

        /*
         * Create an optimizer and allow it to tweak the weights and biases.
         */
        AdamOptimizer optimizer = new AdamOptimizer()
                .tweak(layer1Weights)
                .tweak(layer1Biases)
                .tweak(layer2Weights)
                .tweak(layer2Biases);
        optimizer.learningRate = 0.1f;

        /*
         * Create the deltas and use them to minimize 'cost'.
         */
        HashMap<VariableNode, Node> deltas = new HashMap<>();
        cost.createGradients(deltas, null);
        optimizer.minimize(cost, deltas);

        /*
         * Create the inputs and target outputs for the network.
         */
        Tensor xVal = new Tensor(new float[]{
                0, 0,
                0, 1,
                1, 0,
                1, 1
        }, new int[]{4, 2});
        Tensor yTargetVal = new Tensor(new float[]{
                0,
                1,
                1,
                0
        }, new int[]{4, 1});

        /*
         * Run for 100 iterations.
         */
        for (int i = 0; i < 100; i++) {
            /*
             * Feed the data into the optimizer.
             */
            Eval eval = new Eval()
                    .addInput(x, xVal)
                    .addInput(yTarget, yTargetVal);
            optimizer.run(eval);

            /*
             * Print the error every 10 iterations.
             */
            if (i % 10 == 0) {
                System.out.printf("Error: %f\n", eval.evaluate(cost).getVal(0));
            }
        }

        System.out.println();

        {
            /*
             * Show the total error.
             */
            Eval eval = new Eval()
                    .addInput(x, xVal)
                    .addInput(yTarget, yTargetVal);
            System.out.printf("Total Error: %f%n%n", eval.evaluate(cost).getVal(0));
        }

        /*
         * Display the network's output.
         */
        for (float[] vals : new float[][]{{0, 0}, {0, 1}, {1, 0}, {1, 1}}) {
            Eval eval = new Eval()
                    .addInput(x, new Tensor(vals, new int[]{1, 2}));

            System.out.println(Arrays.toString(vals) + " -> " + eval.evaluate(y).arrayToString());
        }
    }
}
