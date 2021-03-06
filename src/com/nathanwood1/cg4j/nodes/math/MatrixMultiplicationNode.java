package com.nathanwood1.cg4j.nodes.math;

import com.nathanwood1.cg4j.Eval;
import com.nathanwood1.cg4j.Tensor;
import com.nathanwood1.cg4j.exception.IllegalShapeException;
import com.nathanwood1.cg4j.nodes.Node;
import com.nathanwood1.cg4j.nodes.io.VariableNode;
import com.nathanwood1.cg4j.optimizers.Optimizer;

import java.util.Arrays;
import java.util.HashMap;

public class MatrixMultiplicationNode extends Node {
    private final boolean aTransposed;
    private final boolean bTransposed;

    public MatrixMultiplicationNode(String name, Node childA, Node childB) {
        super(FormatShape(childA.shape, childB.shape), name, childA, childB);
        this.aTransposed = false;
        this.bTransposed = false;
    }

    public MatrixMultiplicationNode(Node childA, Node childB) {
        super(FormatShape(childA.shape, childB.shape), null, childA, childB);
        this.aTransposed = false;
        this.bTransposed = false;
    }

    public MatrixMultiplicationNode(String name, Node childA, Node childB,
                                    boolean aTransposed, boolean bTransposed) {
        super(FormatShape(childA.shape, childB.shape, aTransposed, bTransposed), name, childA, childB);
        this.aTransposed = aTransposed;
        this.bTransposed = bTransposed;
    }

    public MatrixMultiplicationNode(Node childA, Node childB,
                                    boolean aTransposed, boolean bTransposed) {
        super(FormatShape(childA.shape, childB.shape, aTransposed, bTransposed), null, childA, childB);
        this.aTransposed = aTransposed;
        this.bTransposed = bTransposed;
    }

    private static int[] FormatShape(int[] aShape, int[] bShape) {
        if (aShape.length < 2) {
            throw new IllegalShapeException(
                    "Cannot matrix multiply shapes ("
                            + Arrays.toString(aShape)
                            + ", "
                            + Arrays.toString(bShape)
                            + ")"
            );
        }
        if (bShape.length < 2) {
            throw new IllegalShapeException(
                    "Cannot matrix multiply shapes ("
                            + Arrays.toString(aShape)
                            + ", "
                            + Arrays.toString(bShape)
                            + ")"
            );
        }
        if (aShape[aShape.length - 1] != bShape[bShape.length - 2]) {
            throw new IllegalShapeException(
                    "Cannot matrix multiply shapes ("
                            + Arrays.toString(aShape)
                            + ", "
                            + Arrays.toString(bShape)
                            + ")"
            );
        }
        if (!Node.ShapeEndCompatible(aShape, 2, bShape, 2)) {
            throw new IllegalShapeException(
                    "Cannot matrix multiply shapes ("
                            + Arrays.toString(aShape)
                            + ", "
                            + Arrays.toString(bShape)
                            + ")"
            );
        }

        int[] out = new int[Math.max(aShape.length, bShape.length)];
        if (aShape.length > bShape.length) {
            System.arraycopy(aShape, 0, out, 0, aShape.length - 2);
        } else {
            System.arraycopy(bShape, 0, out, 0, bShape.length - 2);
        }
        out[out.length - 2] = aShape[aShape.length - 2];
        out[out.length - 1] = bShape[bShape.length - 1];
        return out;
    }

    private static int[] FormatShape(int[] aShape, int[] bShape,
                                     boolean aTransposed, boolean bTransposed) {
        int[] out;
        if (!aTransposed && !bTransposed) {
            return FormatShape(aShape, bShape);
        } else if (!aTransposed/* && bTransposed*/) {
            if (aShape.length < 2) {
                throw new IllegalShapeException(
                        "Cannot matrix multiply shapes ("
                                + Arrays.toString(aShape)
                                + ", "
                                + Arrays.toString(bShape)
                                + ")"
                );
            }
            if (bShape.length < 2) {
                throw new IllegalShapeException(
                        "Cannot matrix multiply shapes ("
                                + Arrays.toString(aShape)
                                + ", "
                                + Arrays.toString(bShape)
                                + ")"
                );
            }
            if (aShape[aShape.length - 1] != bShape[bShape.length - 1]) {
                throw new IllegalShapeException(
                        "Cannot matrix multiply shapes ("
                                + Arrays.toString(aShape)
                                + ", "
                                + Arrays.toString(bShape)
                                + ")"
                );
            }
            if (!Node.ShapeEndCompatible(aShape, 2, bShape, 2)) {
                throw new IllegalShapeException(
                        "Cannot matrix multiply shapes ("
                                + Arrays.toString(aShape)
                                + ", "
                                + Arrays.toString(bShape)
                                + ")"
                );
            }

            out = new int[Math.max(aShape.length, bShape.length)];
            if (aShape.length > bShape.length) {
                System.arraycopy(aShape, 0, out, 0, aShape.length - 2);
            } else {
                System.arraycopy(bShape, 0, out, 0, bShape.length - 2);
            }
            out[out.length - 2] = aShape[aShape.length - 2];
            out[out.length - 1] = bShape[bShape.length - 2];
        } else if (/*aTransposed && */!bTransposed) {
            if (aShape.length < 2) {
                throw new IllegalShapeException(
                        "Cannot matrix multiply shapes ("
                                + Arrays.toString(aShape)
                                + ", "
                                + Arrays.toString(bShape)
                                + ")"
                );
            }
            if (bShape.length < 2) {
                throw new IllegalShapeException(
                        "Cannot matrix multiply shapes ("
                                + Arrays.toString(aShape)
                                + ", "
                                + Arrays.toString(bShape)
                                + ")"
                );
            }
            if (aShape[aShape.length - 2] != bShape[bShape.length - 2]) {
                throw new IllegalShapeException(
                        "Cannot matrix multiply shapes ("
                                + Arrays.toString(aShape)
                                + ", "
                                + Arrays.toString(bShape)
                                + ")"
                );
            }
            if (!Node.ShapeEndCompatible(aShape, 2, bShape, 2)) {
                throw new IllegalShapeException(
                        "Cannot matrix multiply shapes ("
                                + Arrays.toString(aShape)
                                + ", "
                                + Arrays.toString(bShape)
                                + ")"
                );
            }

            out = new int[Math.max(aShape.length, bShape.length)];
            if (aShape.length > bShape.length) {
                System.arraycopy(aShape, 0, out, 0, aShape.length - 2);
            } else {
                System.arraycopy(bShape, 0, out, 0, bShape.length - 2);
            }
            out[out.length - 2] = aShape[aShape.length - 1];
            out[out.length - 1] = bShape[bShape.length - 1];
        } else/* if (aTransposed && bTransposed)*/ {
            if (aShape.length < 2) {
                throw new IllegalShapeException(
                        "Cannot matrix multiply shapes ("
                                + Arrays.toString(aShape)
                                + ", "
                                + Arrays.toString(bShape)
                                + ")"
                );
            }
            if (bShape.length < 2) {
                throw new IllegalShapeException(
                        "Cannot matrix multiply shapes ("
                                + Arrays.toString(aShape)
                                + ", "
                                + Arrays.toString(bShape)
                                + ")"
                );
            }
            if (aShape[aShape.length - 2] != bShape[bShape.length - 1]) {
                throw new IllegalShapeException(
                        "Cannot matrix multiply shapes ("
                                + Arrays.toString(aShape)
                                + ", "
                                + Arrays.toString(bShape)
                                + ")"
                );
            }
            if (!Node.ShapeEndCompatible(aShape, 2, bShape, 2)) {
                throw new IllegalShapeException(
                        "Cannot matrix multiply shapes ("
                                + Arrays.toString(aShape)
                                + ", "
                                + Arrays.toString(bShape)
                                + ")"
                );
            }

            out = new int[Math.max(aShape.length, bShape.length)];
            if (aShape.length > bShape.length) {
                System.arraycopy(aShape, 0, out, 0, aShape.length - 2);
            } else {
                System.arraycopy(bShape, 0, out, 0, bShape.length - 2);
            }
            out[out.length - 2] = aShape[aShape.length - 1];
            out[out.length - 1] = bShape[bShape.length - 2];
        }
        return out;
    }

    @Override
    protected String getNodeClassName() {
        return "MatrixMultiplicationNode";
    }

    /**
     * Use {@code Eval#evaluate(Node)}
     *
     * @see Eval#evaluate(Node)
     */
    @Override
    public Tensor evaluate(Eval e) {
        Tensor a = e.evaluate(children[0]);
        Tensor b = e.evaluate(children[1]);

        Tensor out;
        if (!aTransposed && !bTransposed) {
            int[] shape = new int[Math.max(a.shape.length, b.shape.length)];
            if (a.shape.length > b.shape.length) {
                System.arraycopy(a.shape, 0, shape, 0, a.shape.length - 2);
            } else {
                System.arraycopy(b.shape, 0, shape, 0, b.shape.length - 2);
            }
            shape[shape.length - 2] = a.shape[a.shape.length - 2];
            shape[shape.length - 1] = b.shape[b.shape.length - 1];
            int length = 1;
            for (int x : shape) {
                length *= x;
            }
            out = new Tensor(new float[length], shape);
            for (int h = 0; h < out.length / (shape[shape.length - 1] * shape[shape.length - 2]); h++) {
                for (int i = 0; i < a.shape[a.shape.length - 2]; i++) {
                    for (int j = 0; j < b.shape[b.shape.length - 1]; j++) {
                        float val = 0;
                        for (int k = 0; k < a.shape[a.shape.length - 1]; k++) {
                            val += a.getVal((h * (a.shape[a.shape.length - 2] * a.shape[a.shape.length - 1])
                                    + i * a.shape[a.shape.length - 1] + k) % a.length) *
                                    b.getVal((h * (b.shape[b.shape.length - 2] * b.shape[b.shape.length - 1])
                                            + k * b.shape[b.shape.length - 1] + j) % b.length);
                        }
                        out.setVal(h * (shape[shape.length - 1] * shape[shape.length - 2])
                                           + i * out.shape[out.shape.length - 1] + j, val);
                    }
                }
            }
        } else if (!aTransposed && bTransposed) {
            int[] shape = new int[Math.max(a.shape.length, b.shape.length)];
            if (a.shape.length > b.shape.length) {
                System.arraycopy(a.shape, 0, shape, 0, a.shape.length - 2);
            } else {
                System.arraycopy(b.shape, 0, shape, 0, b.shape.length - 2);
            }
            shape[shape.length - 2] = a.shape[a.shape.length - 2];
            shape[shape.length - 1] = b.shape[b.shape.length - /*1*/2];
            int length = 1;
            for (int x : shape) {
                length *= x;
            }
            out = new Tensor(new float[length], shape);
            for (int h = 0; h < out.length / (shape[shape.length - 1] * shape[shape.length - 2]); h++) {
                for (int i = 0; i < a.shape[a.shape.length - 2]; i++) {
                    for (int j = 0; j < /*b.shape[b.shape.length - 1]*/b.shape[b.shape.length - 2]; j++) {
                        float val = 0;
                        for (int k = 0; k < a.shape[a.shape.length - 1]; k++) {
                            val += a.getVal((h * (a.shape[a.shape.length - 2] * a.shape[a.shape.length - 1])
                                    + i * a.shape[a.shape.length - 1] + k) % a.length)
                                    * b.getVal((h * (b.shape[b.shape.length - 1] * b.shape[b.shape.length - 2]))
                                                       + j * b.shape[b.shape.length - 1] + k);
                        }
                        out.setVal(h * (shape[shape.length - 1] * shape[shape.length - 2])
                                           + i * out.shape[out.shape.length - 1] + j, val);
                    }
                }
            }
        } else if (aTransposed && !bTransposed) {
            int[] shape = new int[Math.max(a.shape.length, b.shape.length)];
            if (a.shape.length > b.shape.length) {
                System.arraycopy(a.shape, 0, shape, 0, a.shape.length - 2);
            } else {
                System.arraycopy(b.shape, 0, shape, 0, b.shape.length - 2);
            }
            shape[shape.length - 2] = a.shape[a.shape.length - /*2*/1];
            shape[shape.length - 1] = b.shape[b.shape.length - 1];
            int length = 1;
            for (int x : shape) {
                length *= x;
            }
            out = new Tensor(new float[length], shape);
            for (int h = 0; h < out.length / (shape[shape.length - 1] * shape[shape.length - 2]); h++) {
                for (int i = 0; i < /*a.shape[a.shape.length - 2]*/a.shape[a.shape.length - 1]; i++) {
                    for (int j = 0; j < b.shape[b.shape.length - 1]; j++) {
                        float val = 0;
                        for (int k = 0; k < a.shape[a.shape.length - 2]; k++) {
                            val += a.getVal((h * (a.shape[a.shape.length - 2] * a.shape[a.shape.length - 1])
                                    + k * a.shape[a.shape.length - 1] + i) % a.length) *
                                    b.getVal((h * (b.shape[b.shape.length - 2] * b.shape[b.shape.length - 1])
                                            + k * b.shape[b.shape.length - 1] + j) % b.length);
                        }
                        out.setVal(h * (shape[shape.length - 1] * shape[shape.length - 2])
                                           + i * out.shape[out.shape.length - 1] + j, val);
                    }
                }
            }
        } else {
            int[] shape = new int[Math.max(a.shape.length, b.shape.length)];
            if (a.shape.length > b.shape.length) {
                System.arraycopy(a.shape, 0, shape, 0, a.shape.length - 2);
            } else {
                System.arraycopy(b.shape, 0, shape, 0, b.shape.length - 2);
            }
            shape[shape.length - 2] = a.shape[a.shape.length - /*2*/1];
            shape[shape.length - 1] = b.shape[b.shape.length - /*1*/2];
            int length = 1;
            for (int x : shape) {
                length *= x;
            }
            out = new Tensor(new float[length], shape);
            for (int h = 0; h < out.length / (shape[shape.length - 1] * shape[shape.length - 2]); h++) {
                for (int i = 0; i < /*a.shape[a.shape.length - 2]*/a.shape[a.shape.length - 1]; i++) {
                    for (int j = 0; j < /*b.shape[b.shape.length - 1]*/b.shape[b.shape.length - 2]; j++) {
                        float val = 0;
                        for (int k = 0; k < a.shape[a.shape.length - 2]; k++) {
                            val += a.getVal((h * (a.shape[a.shape.length - 2] * a.shape[a.shape.length - 1])
                                    + k * a.shape[a.shape.length - 1] + i) % a.length) *
                                    b.getVal((h * (b.shape[b.shape.length - 1] * b.shape[b.shape.length - 2]))
                                                     + j * b.shape[b.shape.length - 1] + k);
                        }
                        out.setVal(h * (shape[shape.length - 1] * shape[shape.length - 2])
                                           + i * out.shape[out.shape.length - 1] + j, val);
                    }
                }
            }
        }
        return out;
    }

    @Override
    public void createGradients(HashMap<VariableNode, Node> deltas, Node parentDelta) {
        children[0].createGradients(deltas, new MatrixMultiplicationNode(parentDelta, children[1],
                                                                            false,
                                                                            !bTransposed));
        children[1].createGradients(deltas, new MatrixMultiplicationNode(children[0], parentDelta,
                                                                            !aTransposed,
                                                                            false));
    }
}
