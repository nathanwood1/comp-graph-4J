package com.nathanwood1.cg4j.nodes.io;

import com.nathanwood1.cg4j.Eval;
import com.nathanwood1.cg4j.Tensor;
import com.nathanwood1.cg4j.nodes.Node;
import com.nathanwood1.cg4j.optimizers.Optimizer;

public class OutputNode extends Node {
    public OutputNode(String name, Node child) {
        super(child.shape, name, child);
    }

    public OutputNode(Node child) {
        super(child.shape, null, child);
    }

    @Override
    public String getNodeClassName() {
        return "OutputNode";
    }

    @Override
    public Tensor evaluate(Eval e) {
        return e.evaluate(children[0]);
    }

    @Override
    public void createGradients(Optimizer optimizer, Node parentDelta) {
        children[0].createGradients(optimizer, parentDelta);
    }
}
