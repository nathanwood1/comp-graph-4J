package com.nathanwood1.cg4j.nodes.nn;

import com.nathanwood1.cg4j.Eval;
import com.nathanwood1.cg4j.Tensor;
import com.nathanwood1.cg4j.nodes.Node;
import com.nathanwood1.cg4j.nodes.io.VariableNode;
import com.nathanwood1.cg4j.optimizers.Optimizer;

import java.util.HashMap;

public class ReluNode extends Node {
    private float[] vals;

    public ReluNode(String name, Node child) {
        super(child.shape, name, child);
    }

    public ReluNode(Node child) {
        super(child.shape, null, child);
    }

    @Override
    protected String getNodeClassName() {
        return "ReluNode";
    }

    /**
     * Use {@code Eval#evaluate(Node)}
     *
     * @see Eval#evaluate(Node)
     */
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

    @Override
    public void createGradients(HashMap<VariableNode, Node> deltas, Node parentDelta) {
        children[0].createGradients(deltas, new ReluDeltaNode(parentDelta));
    }

    private class ReluDeltaNode extends Node {
        public ReluDeltaNode(Node child) {
            super(child.shape, null, child);
        }

        @Override
        protected String getNodeClassName() {
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
        public void createGradients(HashMap<VariableNode, Node> deltas, Node parentDelta) {

        }
    }
}
