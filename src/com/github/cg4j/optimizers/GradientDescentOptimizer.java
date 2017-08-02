package com.github.cg4j.optimizers;

import com.github.cg4j.Eval;
import com.github.cg4j.Tensor;
import com.github.cg4j.exception.NoVariableNodeToMinimizeException;
import com.github.cg4j.nodes.Node;
import com.github.cg4j.nodes.io.VariableNode;

public class GradientDescentOptimizer extends Optimizer {
    public Node toMinimize = null;

    public float learningRate = 0.001f;

    @Override
    public void minimize(Node toMinimize) {
        this.toMinimize = toMinimize;
        toMinimize.createGradients(this, toMinimize);
    }

    @Override
    public void run(Eval eval) {
        if (toMinimize == null) {
            throw new NoVariableNodeToMinimizeException("No nodes where told to minimize!");
        }
        for (VariableNode variable : deltas.keySet()) {
            Node delta = deltas.get(variable);
            Tensor variableD = eval.evaluate(delta);
            for (int i = 0; i < variableD.length; i++) {
                variable.val.setVal(i, variable.val.getVal(i) - variableD.getVal(i) * learningRate);
            }
        }
    }
}
