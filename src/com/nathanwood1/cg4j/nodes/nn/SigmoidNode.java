package com.nathanwood1.cg4j.nodes.nn;

import com.nathanwood1.cg4j.Eval;
import com.nathanwood1.cg4j.Tensor;
import com.nathanwood1.cg4j.nodes.Node;
import com.nathanwood1.cg4j.nodes.io.VariableNode;
import com.nathanwood1.cg4j.optimizers.Optimizer;

import java.util.HashMap;

public class SigmoidNode extends Node {
    private float[] vals;

    public SigmoidNode(String name, Node child) {
        super(child.shape, name, child);
    }

    public SigmoidNode(Node child) {
        super(child.shape, null, child);
    }

    @Override
    protected String getNodeClassName() {
        return "SigmoidNode";
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
            vals[i] = 1 / (1 + (float) Math.exp(-in.getVal(i)));
        }

        Tensor out = new Tensor(vals, in.shape);
        return out;
    }

    @Override
    public void createGradients(HashMap<VariableNode, Node> deltas, Node parentDelta) {
        children[0].createGradients(deltas, new SigmoidDeltaNode(parentDelta));
    }

    private class SigmoidDeltaNode extends Node {
        public SigmoidDeltaNode(Node child) {
            super(child.shape, null, child);
        }

        @Override
        protected String getNodeClassName() {
            return "SigmoidDeltaNode";
        }

        @Override
        public Tensor evaluate(Eval e) {
            Tensor in = e.evaluate(children[0]);

            Tensor out = new Tensor(new float[in.length], in.shape);
            for (int i = 0; i < out.length; i++) {
                out.setVal(i, vals[i] * (1 - vals[i]) * in.getVal(i));
            }
            return out;
        }

        @Override
        public void createGradients(HashMap<VariableNode, Node> deltas, Node parentDelta) {
        }
    }
}
