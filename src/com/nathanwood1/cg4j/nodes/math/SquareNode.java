package com.nathanwood1.cg4j.nodes.math;

import com.nathanwood1.cg4j.Eval;
import com.nathanwood1.cg4j.Tensor;
import com.nathanwood1.cg4j.nodes.Node;
import com.nathanwood1.cg4j.nodes.io.ConstantNode;
import com.nathanwood1.cg4j.optimizers.Optimizer;

public class SquareNode extends Node {
    public SquareNode(String name, Node child) {
        super(child.shape, name, child);
    }

    public SquareNode(Node child) {
        super(child.shape, null, child);
    }

    @Override
    public String getNodeClassName() {
        return "SquareNode";
    }

    @Override
    public Tensor evaluate(Eval e) {
        Tensor in = e.evaluate(children[0]);
        Tensor out = new Tensor(new float[in.length], in.shape);
        for (int i = 0; i < out.length; i++) {
            out.setVal(i, in.getVal(i) * in.getVal(i));
        }
        return out;
    }

    @Override
    public void createGradients(Optimizer optimizer, Node parentDelta) {
        MultiplicationNode mult;
        if (parentDelta == null) {
            mult = new MultiplicationNode(
                    children[0],
                    new ConstantNode(
                            new Tensor(
                                    new float[]{2f}
                                    , new int[]{1}
                            )
                    )
            );
        } else {
            mult = new MultiplicationNode(
                    parentDelta,
                    children[0],
                    new ConstantNode(
                            new Tensor(
                                    new float[]{2f}
                                    , new int[]{1}
                            )
                    )
            );
        }
        children[0].createGradients(optimizer, mult);
    }
}
