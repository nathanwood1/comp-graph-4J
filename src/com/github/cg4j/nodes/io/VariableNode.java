package com.github.cg4j.nodes.io;

import com.github.cg4j.Eval;
import com.github.cg4j.Tensor;
import com.github.cg4j.nodes.Node;
import com.github.cg4j.optimizers.Optimizer;

public class VariableNode extends Node {
    public final Tensor val;

    public VariableNode(Tensor val) {
        super(val.getShape(), null);
        this.val = val;
    }

    public VariableNode(Tensor val, String name) {
        super(val.getShape(), name);
        this.val = val;
    }

    @Override
    public String getNodeClassName() {
        return "VariableNode";
    }

    @Override
    protected boolean canAddChildren() {
        return false;
    }

    @Override
    public Tensor evaluate(Eval e) {
        return val;
    }

    private boolean gradientCreated = false;

    @Override
    public void createGradients(Optimizer optimizer, Node parentDelta) {
        if (gradientCreated) {
            optimizer.deltas.get(this).addChild(parentDelta);
        } else {
            optimizer.put(this, new VariableDeltaNode(parentDelta));
            gradientCreated = true;
        }
    }
}
