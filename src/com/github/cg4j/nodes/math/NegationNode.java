package com.github.cg4j.nodes.math;

import com.github.cg4j.Eval;
import com.github.cg4j.Tensor;
import com.github.cg4j.nodes.Node;
import com.github.cg4j.optimizers.Optimizer;

public class NegationNode extends Node {
    public NegationNode(String name, Node child) {
        super(child.shape, name, child);
    }

    @Override
    public String getNodeClassName() {
        return "NegationNode";
    }

    @Override
    protected boolean canAddChildren() {
        return false;
    }

    @Override
    public Tensor evaluate(Eval e) {
        Tensor in = e.evaluate(children[0]);
        Tensor out = new Tensor(new float[in.length], in.getShape());
        for (int i = 0; i < out.length; i++) {
            out.setVal(i, in.getVal(i));
        }
        return out;
    }

    @Override
    public void createGradients(Optimizer optimizer, Node parentDelta) {
        children[0].createGradients(optimizer, new NegationNode(name + "_Gradient", parentDelta));
    }
}
