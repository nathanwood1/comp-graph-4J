package com.github.cg4j.nodes.math;

import com.github.cg4j.Eval;
import com.github.cg4j.Tensor;
import com.github.cg4j.exception.IllegalShapeException;
import com.github.cg4j.nodes.Node;

public abstract class MathNode extends Node {

    public MathNode(int[] shape, String name, Node... children) {
        super(shape, name, children);
        if (children.length == 0) {
            throw new IllegalShapeException("A math node must have at least one child!");
        }
    }

    @Override
    public Tensor evaluate(Eval e) {
        Tensor[] evals = new Tensor[children.length];
        for (int i = 0; i < children.length; i++) {
            evals[i] = e.evaluate(children[i]);
        }
    }

    protected abstract float evaluate(float[] children);
}
