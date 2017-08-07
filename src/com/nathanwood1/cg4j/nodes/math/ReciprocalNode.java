package com.nathanwood1.cg4j.nodes.math;

import com.nathanwood1.cg4j.Eval;
import com.nathanwood1.cg4j.Tensor;
import com.nathanwood1.cg4j.nodes.Node;
import com.nathanwood1.cg4j.nodes.io.VariableNode;

import java.util.HashMap;

public class ReciprocalNode extends Node {
    public ReciprocalNode(String name, Node child) {
        super(child.shape, name, child);
    }

    public ReciprocalNode(Node child) {
        super(child.shape, null, child);
    }

    @Override
    protected String getNodeClassName() {
        return "ReciprocalNode";
    }

    @Override
    public Tensor evaluate(Eval e) {
        Tensor in = e.evaluate(children[0]);
        Tensor out = new Tensor(new float[in.length], in.shape);
        for (int i = 0; i < out.length; i++) {
            out.setVal(i, 1 / in.getVal(i));
        }
        return out;
    }

    @Override
    public void createGradients(HashMap<VariableNode, Node> deltas, Node parentDelta) {
        Node delta = new NegationNode(new ReciprocalNode(new SquareNode(parentDelta)));
        children[0].createGradients(deltas, delta);
    }
}
