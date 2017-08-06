package com.nathanwood1.cg4j.examples;

import com.nathanwood1.cg4j.Eval;
import com.nathanwood1.cg4j.Tensor;
import com.nathanwood1.cg4j.nodes.Node;
import com.nathanwood1.cg4j.nodes.io.InputNode;
import com.nathanwood1.cg4j.nodes.io.OutputNode;
import com.nathanwood1.cg4j.nodes.io.VariableNode;
import com.nathanwood1.cg4j.nodes.math.AdditionNode;
import com.nathanwood1.cg4j.nodes.math.MatrixMultiplicationNode;
import com.nathanwood1.cg4j.nodes.math.SquareNode;
import com.nathanwood1.cg4j.nodes.math.SubtractionNode;
import com.nathanwood1.cg4j.nodes.nn.ReluNode;
import com.nathanwood1.cg4j.nodes.nn.SigmoidNode;
import com.nathanwood1.cg4j.nodes.tensor.MeanNode;
import com.nathanwood1.cg4j.optimizers.GradientDescentOptimizer;

import java.util.Arrays;
import java.util.Random;

public class XOR {

    public static void main(String[] args) {
        Random seed = new Random(0);

        VariableNode layer1Weights = new VariableNode(Tensor.fromRandom(seed, -1, 1, new int[]{2, 4}));
        VariableNode layer1Biases = new VariableNode(Tensor.fromRandom(seed, -1, 1, new int[]{4}));
        VariableNode layer2Weights = new VariableNode(Tensor.fromRandom(seed, -1, 1, new int[]{4, 1}));
        VariableNode layer2Biases = new VariableNode(Tensor.fromRandom(seed, -1, 1, new int[]{1}));

        InputNode x = new InputNode(new int[]{-1, 2});

        Node y = new MatrixMultiplicationNode(x, layer1Weights);
        y = new AdditionNode(y, layer1Biases);
        y = new SigmoidNode(y);

        y = new MatrixMultiplicationNode(y, layer2Weights);
        y = new AdditionNode(y, layer2Biases);
        y = new SigmoidNode(y);

        OutputNode yOut = new OutputNode(y);

        InputNode yTarget = new InputNode(new int[]{-1, 1});
        OutputNode cost = new OutputNode(new MeanNode(new SquareNode(new SubtractionNode(yTarget,
                                                                                         y))));
        System.out.println(cost);

        GradientDescentOptimizer optimizer = new GradientDescentOptimizer()
                .tweak(layer1Weights)
                .tweak(layer1Biases)
                .tweak(layer2Weights)
                .tweak(layer2Biases);
        optimizer.learningRate = 1f;
        optimizer.minimize(cost);

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

        for (int i = 0; i < 2000; i++) {
            Eval eval = new Eval()
                    .addInput(x, xVal)
                    .addInput(yTarget, yTargetVal);
            optimizer.run(eval);

            if (i % 100 == 0) {
                System.out.printf("Error: %f\n", eval.evaluate(cost).getVal(0));
            }
        }

        System.out.println();

        {
            Eval eval = new Eval()
                    .addInput(x, xVal)
                    .addInput(yTarget, yTargetVal);
            System.out.printf("Total Error: %f\n\n", eval.evaluate(cost).getVal(0));
        }

        for (float[] vals : new float[][]{{0, 0}, {0, 1}, {1, 0}, {1, 1}}) {
            Eval eval = new Eval()
                    .addInput(x, new Tensor(vals, new int[]{1, 2}));

            System.out.println(Arrays.toString(vals) + " -> " + eval.evaluate(yOut).arrayToString());
        }
    }
}
