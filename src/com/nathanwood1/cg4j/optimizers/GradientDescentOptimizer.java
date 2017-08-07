package com.nathanwood1.cg4j.optimizers;

import com.nathanwood1.cg4j.Eval;
import com.nathanwood1.cg4j.Tensor;
import com.nathanwood1.cg4j.exception.NoVariableNodeToMinimizeException;
import com.nathanwood1.cg4j.nodes.Node;
import com.nathanwood1.cg4j.nodes.io.VariableNode;

import java.util.HashMap;
import java.util.LinkedList;

public class GradientDescentOptimizer extends Optimizer {
    private LinkedList<VariableNode> toTweak;

    private Node toChange = null;
    private boolean minimizing;

    public float learningRate = 0.001f;

    public GradientDescentOptimizer() {
        toTweak = new LinkedList<>();
    }

    public GradientDescentOptimizer tweak(VariableNode toTweak) {
        this.toTweak.add(toTweak);
        return this;
    }

    @Override
    public void minimize(Node toChange, HashMap<VariableNode, Node> deltas) {
        this.toChange = toChange;
        minimizing = true;
        this.deltas = deltas;
    }

    @Override
    public void maximize(Node toChange, HashMap<VariableNode, Node> deltas) {
        this.toChange = toChange;
        minimizing = false;
        this.deltas = deltas;
    }

    @Override
    public void run(Eval eval) {
        if (toChange == null) {
            throw new NoVariableNodeToMinimizeException("No nodes where told to minimize!");
        }
        if (minimizing) {
            for (VariableNode variable : toTweak) {
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
        } else {
            for (VariableNode variable : toTweak) {
                Node delta = deltas.get(variable);
                Tensor variableD = eval.evaluate(delta);
                for (int i = 0; i < variableD.length; i++) {
                    variable.val.setVal(i % variable.val.length,
                                        variable.val.getVal(i % variable.val.length)
                                                + variableD.getVal(i)
                                                * learningRate
                    );
                }
            }
        }
    }
}
