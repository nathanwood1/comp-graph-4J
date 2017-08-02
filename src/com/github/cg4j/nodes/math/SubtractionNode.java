package com.github.cg4j.nodes.math;

import com.github.cg4j.Eval;
import com.github.cg4j.Tensor;
import com.github.cg4j.exception.IllegalShapeException;
import com.github.cg4j.nodes.Node;
import com.github.cg4j.optimizers.Optimizer;

import java.util.Arrays;

public class SubtractionNode extends Node {
    public SubtractionNode(String name, Node a, Node b) {
        super(a.shape, name, a, b);
        if (!Node.ShapeEndCompatible(a.shape, 0, b.shape,0)) {
            throw new IllegalShapeException(
                    "Ends of the shapes must be equal ("
                            + Arrays.toString(a.shape)
                            + ", "
                            + Arrays.toString(b.shape)
                            + ")"
            );
        }
    }

    public SubtractionNode(Node a, Node b) {
        super(a.shape, null, a, b);
        if (!Node.ShapeEndCompatible(a.shape, 0, b.shape,0)) {
            throw new IllegalShapeException(
                    "Ends of the shapes must be equal ("
                            + Arrays.toString(a.shape)
                            + ", "
                            + Arrays.toString(b.shape)
                            + ")"
            );
        }
    }

    @Override
    public String getNodeClassName() {
        return "SubtractionNode";
    }

    @Override
    protected boolean canAddChildren() {
        return false;
    }

    @Override
    public Tensor evaluate(Eval e) {
        Tensor aIn = e.evaluate(children[0]);
        Tensor bIn = e.evaluate(children[1]);
        Tensor out = new Tensor(new float[aIn.length], aIn.getShape());
        for (int i = 0; i < out.length; i++) {
            out.setVal(i, aIn.getVal(i) - bIn.getVal(i));
        }
        return out;
    }

    @Override
    public void createGradients(Optimizer optimizer, Node parentDelta) {
        children[0].createGradients(optimizer, parentDelta);
        children[1].createGradients(optimizer, new NegationNode(name + "_Gradient", parentDelta));
    }
}
