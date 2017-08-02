package com.github.cg4j.examples;

import com.github.cg4j.Eval;
import com.github.cg4j.Tensor;
import com.github.cg4j.nodes.Node;
import com.github.cg4j.nodes.io.InputNode;
import com.github.cg4j.nodes.io.OutputNode;
import com.github.cg4j.nodes.io.VariableNode;
import com.github.cg4j.nodes.math.AdditionNode;
import com.github.cg4j.nodes.math.MeanNode;
import com.github.cg4j.nodes.math.MultiplicationNode;
import com.github.cg4j.nodes.math.SquareNode;
import com.github.cg4j.nodes.math.SubtractionNode;
import com.github.cg4j.optimizers.GradientDescentOptimizer;
import com.github.cg4j.optimizers.Optimizer;

import java.io.IOException;
import java.net.URL;
import java.util.Scanner;

public class LinearRegression {

    public static void main(String[] args) {
        VariableNode m = new VariableNode(new Tensor(new float[]{3f}, new int[]{1}));
        VariableNode c = new VariableNode(new Tensor(new float[]{-1f}, new int[]{1}));

        InputNode x = new InputNode(new int[]{-1, 1});
        x.evaluate(null);

        Node y = new MultiplicationNode(x, m);
        y = new AdditionNode(y, c);
        OutputNode yOut = new OutputNode(y);

        InputNode yTarget = new InputNode(new int[]{-1, 1});
        Node cost = new MeanNode(new SquareNode(new SubtractionNode(yTarget, y)));

        GradientDescentOptimizer optimizer = new GradientDescentOptimizer();
        optimizer.minimize(cost);

        Tensor xInputData = new Tensor(new float[]{
                1,
                3
        }, new int[]{2, 1});
        Tensor yTargetInputData = new Tensor(new float[]{
                3,
                6
        }, new int[]{2, 1});

        float learningRate = 0.001f;

        for (int i = 0; i < 100000; i++) {
            Eval eval = new Eval()
                    .addInput(x, xInputData)
                    .addInput(yTarget, yTargetInputData);

            optimizer.run(eval);
        }
    }

    public static Tensor getData() {
        StringBuilder str = new StringBuilder();
        // Data from Siraj (https://github.com/llSourcell/)
        try (Scanner s = new Scanner(new URL("https://raw.githubusercontent.com/llSourcell/linear_regression_live/master/data.csv").openStream())) {
            while (s.hasNextLine()) {
                str.append(s.nextLine()).append("\n");
            }
        } catch (IOException e) {
            e.printStackTrace();
            System.exit(1);
        }
        str.deleteCharAt(str.length() - 1); //Remove the last '\n'
        String[] nLineSplit = str.toString().split("\n");
        float[] data = new float[nLineSplit.length * 2];
        for (int i = 0; i < nLineSplit.length; i++) {
            String[] thisSplit = nLineSplit[i].split(",");
            data[i * 2] = Float.parseFloat(thisSplit[0]);
            data[i * 2 + 1] = Float.parseFloat(thisSplit[1]);
        }
        return new Tensor(data, new int[]{nLineSplit.length, 2});
    }
}
