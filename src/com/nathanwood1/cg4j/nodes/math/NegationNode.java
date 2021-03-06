package com.nathanwood1.cg4j.nodes.math;

import com.nathanwood1.cg4j.Eval;
import com.nathanwood1.cg4j.Tensor;
import com.nathanwood1.cg4j.nodes.Node;
import com.nathanwood1.cg4j.nodes.io.VariableNode;
import com.nathanwood1.cg4j.optimizers.Optimizer;

import java.util.HashMap;

public class NegationNode extends Node {
    public NegationNode(String name, Node child) {
        super(child.shape, name, child);
    }

    public NegationNode(Node child) {
        super(child.shape, null, child);
    }

    @Override
    protected String getNodeClassName() {
        return "NegationNode";
    }

    /**
     * Use {@code Eval#evaluate(Node)}
     *
     * @see Eval#evaluate(Node)
     */
    @Override
    public Tensor evaluate(Eval e) {
        Tensor in = e.evaluate(children[0]);
        Tensor out = new Tensor(new float[in.length], in.shape);
        for (int i = 0; i < out.length; i++) {
            out.setVal(i, -in.getVal(i));
        }
        return out;
    }

    @Override
    public void createGradients(HashMap<VariableNode, Node> deltas, Node parentDelta) {
        children[0].createGradients(deltas, new NegationNode(name + "_Gradient", parentDelta));
    }
}
