package com.github.cg4j.nodes;

import com.github.cg4j.Eval;
import com.github.cg4j.Tensor;
import com.github.cg4j.optimizers.Optimizer;

import java.util.Arrays;

//TODO: ADD JAVADOC FOR ALL NODES
public abstract class Node {
    private static int stringGlobalCounter = 0;

    public final String name;
    public final int[] shape;

    protected Node[] children;

    public Node(int[] shape, String name, Node... children) {
        this.children = children;
        this.shape = shape;
        if (name == null) {
            this.name = nodeClassName() + "_" + stringGlobalCounter++;
        } else {
            this.name = name;
        }
    }

    public abstract String nodeClassName();

    public Node[] getChildren() {
        return children;
    }

    public abstract Tensor evaluate(Eval e);

    public abstract void createGradients(Optimizer optimizer, Node parentDelta);

    @Override
    public String toString() {
        return "(" + name + ", " + Arrays.toString(children) + ")";
    }

    public static boolean ShapeEndCompatible(int[] aShape, int aStart, int[] bShape, int bStart) {
        for (int i = 1; i <= Math.min(aShape.length - bStart, bShape.length - aStart); i++) {
            if (aShape[aShape.length - i - bStart] != bShape[bShape.length - i - aStart]) {
                return false;
            }
        }
        return true;
    }
}
