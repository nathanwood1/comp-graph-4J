package com.nathanwood1.cg4j.nodes.nn;

import com.nathanwood1.cg4j.Eval;
import com.nathanwood1.cg4j.Tensor;
import com.nathanwood1.cg4j.nodes.Node;
import com.nathanwood1.cg4j.optimizers.Optimizer;

public class ReluNode extends Node {
    public ReluNode(String name, Node child) {
        super(child.shape, name, child);
    }

    public ReluNode(Node child) {
        super(child.shape, null, child);
    }

    @Override
    public String getNodeClassName() {
        return "ReluNode";
    }

    @Override
    public Tensor evaluate(Eval e) {
        Tensor in = e.evaluate(children[0]);

        vals = new float[in.length];
        for (int i = 0; i < vals.length; i++) {
            vals[i] = in.getVal(i) < 0 ? 0 : in.getVal(i);
        }
        Tensor out = new Tensor(vals, in.shape);
        return out;
    }

    private float[] vals;

    @Override
    public void createGradients(Optimizer optimizer, Node parentDelta) {
        children[0].createGradients(optimizer, new ReluDeltaNode(parentDelta));
    }

    private class ReluDeltaNode extends Node {
        public ReluDeltaNode(Node child) {
            super(child.shape, null, child);
        }

        @Override
        public String getNodeClassName() {
            return "ReluDeltaNode";
        }

        @Override
        public Tensor evaluate(Eval e) {
            Tensor in = e.evaluate(children[0]);

            Tensor out = new Tensor(new float[in.length], in.shape);
            for (int i = 0; i < out.length; i++) {
                out.setVal(i, vals[i] == 0 ? 0 : in.getVal(i));
            }
            return out;
        }

        @Override
        public void createGradients(Optimizer optimizer, Node parentDelta) {

        }
    }
}
