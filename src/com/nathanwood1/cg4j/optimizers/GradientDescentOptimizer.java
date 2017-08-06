package com.nathanwood1.cg4j.optimizers;

import com.nathanwood1.cg4j.Eval;
import com.nathanwood1.cg4j.Tensor;
import com.nathanwood1.cg4j.exception.NoVariableNodeToMinimizeException;
import com.nathanwood1.cg4j.nodes.Node;
import com.nathanwood1.cg4j.nodes.io.VariableNode;

public class GradientDescentOptimizer extends Optimizer {
    public Node toMinimize = null;

    public float learningRate = 0.001f;

    public GradientDescentOptimizer tweak(VariableNode toTweak) {
        deltas.put(toTweak, null);
        return this;
    }

    @Override
    public void minimize(Node toMinimize) {
        this.toMinimize = toMinimize;
        toMinimize.createGradients(this, null);
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
                variable.val.setVal(i % variable.val.length,
                                    variable.val.getVal(i % variable.val.length)
                                            - variableD.getVal(i)
                                            * learningRate
                );
            }
        }
    }
}
