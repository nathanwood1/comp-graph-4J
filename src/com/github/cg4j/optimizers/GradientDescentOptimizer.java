package com.github.cg4j.optimizers;

import com.github.cg4j.Eval;
import com.github.cg4j.exception.NoMinimizeException;
import com.github.cg4j.nodes.Node;
import com.github.cg4j.nodes.io.VariableNode;

public class GradientDescentOptimizer extends Optimizer {
    public Node toMinimize = null;

    @Override
    public void minimize(Node toMinimize) {
        this.toMinimize = toMinimize;
        toMinimize.createGradients(this, toMinimize);
    }

    @Override
    public void run(Eval eval) {
        if (toMinimize == null) {
            throw new NoMinimizeException("No nodes where told to minimize!");
        }
        for (VariableNode key : deltas.keySet()) {
            Node delta = deltas.get(key);
        }
    }
}
