package com.github.cg4j.nodes.io;

import com.github.cg4j.Eval;
import com.github.cg4j.Tensor;
import com.github.cg4j.nodes.Node;
import com.github.cg4j.optimizers.Optimizer;

public class Constant extends Node {
    private final Tensor val;

    public Constant(String name, Tensor val) {
        super(val.getShape(), name);
        this.val = val;
    }

    public Constant(Tensor val) {
        super(val.getShape(), null);
        this.val = val;
    }

    @Override
    public String getNodeClassName() {
        return "Constant";
    }

    @Override
    protected boolean canAddChildren() {
        return false;
    }

    @Override
    public Tensor evaluate(Eval e) {
        return val;
    }

    @Override
    public void createGradients(Optimizer optimizer, Node parentDelta) {
    }
}
