package com.github.cg4j.nodes.math;

import com.github.cg4j.nodes.Node;
import com.github.cg4j.optimizers.Optimizer;

public class AdditionNode extends MathNode {
    public AdditionNode(String name, Node... children) {
        super(children[0].shape, name, children);
    }

    @Override
    protected float evaluate(float[] children) {
        float out = 0;
        for (float c : children) {
            out += c;
        }
        return out;
    }

    @Override
    public String nodeClassName() {
        return "Addition";
    }

    @Override
    public void createGradients(Optimizer optimizer, Node parentDelta) {
        for (Node node : children) {
            node.createGradients(optimizer, parentDelta);
        }
    }
}
