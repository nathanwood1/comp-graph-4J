package com.nathanwood1.cg4j.nodes.math;

import com.nathanwood1.cg4j.Eval;
import com.nathanwood1.cg4j.Tensor;
import com.nathanwood1.cg4j.exception.IllegalShapeException;
import com.nathanwood1.cg4j.nodes.Node;
import com.nathanwood1.cg4j.nodes.io.VariableNode;
import com.nathanwood1.cg4j.optimizers.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

public class MultiplicationNode extends Node {
    public MultiplicationNode(String name, Node... children) {
        super(children[0].shape, name, children);
        for (int i = 1; i < children.length; i++) {
            if (!Arrays.equals(new int[]{1}, children[i].shape))
                if (!Node.ShapeEndCompatible(children[0].shape, 0, children[i].shape, 0)) {
                    throw new IllegalShapeException(
                            "Cannot multiply shapes ("
                                    + Arrays.toString(children[0].shape)
                                    + ", "
                                    + Arrays.toString(children[i].shape)
                                    + ")"
                    );
                }
        }
    }

    public MultiplicationNode(Node... children) {
        super(children[0].shape, null, children);
        for (int i = 1; i < children.length; i++) {
            if (!Arrays.equals(new int[]{1}, children[i].shape))
                if (!Node.ShapeEndCompatible(children[0].shape, 0, children[i].shape, 0)) {
                    throw new IllegalShapeException(
                            "Cannot multiply shapes ("
                                    + Arrays.toString(children[0].shape)
                                    + ", "
                                    + Arrays.toString(children[i].shape)
                                    + ")"
                    );
                }
        }
    }

    @Override
    protected String getNodeClassName() {
        return "MultiplicationNode";
    }

    /**
     * Use {@code Eval#evaluate(Node)}
     *
     * @see Eval#evaluate(Node)
     */
    @Override
    public Tensor evaluate(Eval e) {
        if (children.length == 1) {
            return e.evaluate(children[0]);
        }
        Tensor out = null;
        boolean init = false;
        for (Node child : children) {
            Tensor in = e.evaluate(child);
            if (!init) {
                out = new Tensor(new float[in.length], in.shape);
            }
            for (int i = 0; i < out.length; i++) {
                if (init) {
                    out.setVal(i, out.getVal(i) * in.getVal(i % in.length));
                } else {
                    out.setVal(i, in.getVal(i % in.length));
                }
            }
            init = true;
        }
        return out;
    }

    @Override
    public void createGradients(HashMap<VariableNode, Node> deltas, Node parentDelta) {
        for (Node child : children) {
            ArrayList<Node> multToAdd = new ArrayList<>();
            for (Node childJ : children) {
                if (child != childJ) {
                    multToAdd.add(childJ);
                }
            }
            multToAdd.add(parentDelta);
            MultiplicationNode m = new MultiplicationNode(multToAdd.toArray(new Node[multToAdd.size()]));
            child.createGradients(deltas, m);
        }
    }
}
