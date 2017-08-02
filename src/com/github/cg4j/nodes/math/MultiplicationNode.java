package com.github.cg4j.nodes.math;

import com.github.cg4j.Eval;
import com.github.cg4j.Tensor;
import com.github.cg4j.nodes.Node;
import com.github.cg4j.optimizers.Optimizer;

public class MultiplicationNode extends Node {
    public MultiplicationNode(String name, Node... children) {
        super(children[0].shape, name, children);
    }

    public MultiplicationNode(Node... children) {
        super(children[0].shape, null, children);
    }

    @Override
    public String getNodeClassName() {
        return "MultiplicationNode";
    }

    @Override
    protected boolean canAddChildren() {
        return true;
    }

    @Override
    public Tensor evaluate(Eval e) {
        Tensor out = new Tensor(new float[children[0].length], children[0].shape);
        boolean init = false;
        for (Node child : children) {
            Tensor in = e.evaluate(child);
            for (int i = 0; i < out.length; i++) {
                if (init) {
                    out.setVal(i, out.getVal(i) * in.getVal(i));
                } else {
                    out.setVal(i, in.getVal(i));
                }
            }
        }
        return out;
    }

    @Override
    public void createGradients(Optimizer optimizer, Node parentDelta) {
        for (Node childI : children) {
            MultiplicationNode m = new MultiplicationNode();
            for (Node childJ : children) {
                if (childI != childJ) {
                    m.addChild(childJ);
                }
            }
            childI.createGradients(optimizer, m);
        }
    }
}
