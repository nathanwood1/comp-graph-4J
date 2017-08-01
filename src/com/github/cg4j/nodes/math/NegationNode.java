package com.github.cg4j.nodes.math;

import com.github.cg4j.nodes.Node;
import com.github.cg4j.optimizers.Optimizer;

public class NegationNode extends MathNode {
    public NegationNode(String name, Node child) {
        super(child.shape, name, child);
    }

    @Override
    protected float evaluate(float[] children) {
        return -children[0];
    }

    @Override
    public String nodeClassName() {
        return "NegationNode";
    }

    @Override
    public void createGradients(Optimizer optimizer, Node parentDelta) {
        children[0].createGradients(optimizer, new NegationNode(name + "_gradient", parentDelta));
    }
}
