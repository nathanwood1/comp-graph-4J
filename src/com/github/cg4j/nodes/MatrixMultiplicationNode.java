package com.github.cg4j.nodes;

import com.github.cg4j.Eval;
import com.github.cg4j.Tensor;
import com.github.cg4j.exception.IllegalShapeException;

import java.util.Arrays;

/**
 * A node for multiplying matrices.
 * @since 1.0
 * @author nathanwood1
 */
public class MatrixMultiplicationNode implements Node {
    private final Node childA;
    private final Node childB;

    private final int[] shape;

    /**
     * Create a {@code MatrixMultiplicationNode} from two children nodes.
     * @param a The first child.
     * @param b The second child.
     * @throws IllegalShapeException if the matrices cannot be multiplied because of shape.
     * @since 1.0
     */
    public MatrixMultiplicationNode(Node a, Node b) {
        int[] aShape = a.getShape();
        int[] bShape = b.getShape();
        if (aShape.length <= 1) {
            throw new IllegalShapeException(
                    "Cannot matrix multiply shapes "
                            + Arrays.toString(aShape)
                            + " and "
                            + Arrays.toString(bShape)
                            + " because aShape.length <= 1"
            );
        }
        if (bShape.length <= 1) {
            throw new IllegalShapeException(
                    "Cannot matrix multiply shapes "
                            + Arrays.toString(aShape)
                            + " and "
                            + Arrays.toString(bShape)
                            + " because bShape.length <= 1"
            );
        }
        if (aShape[aShape.length - 1] != bShape[bShape.length - 2]) {
            throw new IllegalShapeException(
                    "Cannot matrix multiply shapes "
                            + Arrays.toString(aShape)
                            + " and "
                            + Arrays.toString(bShape)
                    + " because aShape[-1] != bShape[-2] ("
                    + aShape[aShape.length - 1]
                    + " != "
                    + bShape[bShape.length - 2]
                    + ")"
            );
        }

        shape = new int[aShape.length + bShape.length - 2];
        System.arraycopy(aShape, 0, shape, 0, aShape.length - 1);
        System.arraycopy(bShape, 0, shape, aShape.length - 1, bShape.length - 2);
        shape[shape.length - 1] = bShape[bShape.length - 1];

        this.childA = a;
        this.childB = b;
    }

    /**
     * Returns the shape of this node.
     * @return {@code int[]} shape.
     * @since 1.0
     */
    @Override
    public int[] getShape() {
        return shape;
    }

    /**
     * Returns a list of children of this node.
     * @return {@code Node[]} children.
     * @since 1.0
     */
    @Override
    public Node[] getChildren() {
        return new Node[]{childA, childB};
    }

    /**
     * Calculates the matrix multiplication.
     * Do not use this method; use {@code Eval#evaluate} instead.
     * @see Eval
     * @param e The evaluator.
     * @return The output of the operation.
     * @since 1.0
     */
    @Override
    public Tensor evaluate(Eval e) {
        Tensor a = e.evaluate(childA);
        Tensor b = e.evaluate(childB);

        int[] shape = new int[a.getShape().length + b.getShape().length - 2];
        System.arraycopy(a.getShape(), 0, shape, 0, a.getShape().length - 1);
        System.arraycopy(b.getShape(), 0, shape, a.getShape().length - 1, b.getShape().length - 2);
        shape[shape.length - 1] = b.getShape()[b.getShape().length - 1];

        int length = 1;
        for (int x : shape) {
            length *= x;
        }

        Tensor out = new Tensor(new float[length], shape);

        int iIter = a.getShape()[a.getShape().length - 1];
        int jIter = b.getShape()[b.getShape().length - 1] * b.getShape()[b.getShape().length - 2];
        for (int i = 0; i < a.length; i += iIter) {
            for (int j1 = 0; j1 < b.length; j1 += jIter) {
                for (int j2 = 0; j2 < b.getShape()[b.getShape().length - 1]; j2++) {
                    int j = j1 + j2;
                    int outI = (i / iIter) * (b.length / b.getShape()[b.getShape().length - 2]) + j1 / b.getShape()[b.getShape().length - 2] + j2;
                    for (int k = 0; k < iIter; k++) {
                        out.setVal(outI, out.getVal(outI)
                                + a.getVal(i + k)
                                * b.getVal(j + k * b.getShape()[b.getShape().length - 1])
                        );
                    }
                }
            }
        }
        return out;
    }
}
