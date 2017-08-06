package com.nathanwood1.cg4j.nodes.io;

import com.nathanwood1.cg4j.Eval;
import com.nathanwood1.cg4j.Tensor;
import com.nathanwood1.cg4j.nodes.Node;
import com.nathanwood1.cg4j.optimizers.Optimizer;

public class ConstantNode extends Node {
    private final Tensor val;

    public ConstantNode(String name, Tensor val) {
        super(val.shape, name);
        this.val = val;
    }

    public ConstantNode(Tensor val) {
        super(val.shape, null);
        this.val = val;
    }

    @Override
    public String getNodeClassName() {
        return "ConstantNode";
    }

    @Override
    public Tensor evaluate(Eval e) {
        return val;
    }

    @Override
    public void createGradients(Optimizer optimizer, Node parentDelta) {
    }

    @Override
    public String toString() {
        return val.arrayToString();
    }
}
