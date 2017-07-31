package com.github.cg4j;

import java.util.Arrays;
import java.util.Random;

/**
 * The Tensor class represents an N-Dimensional Array, also called a tensor.
 * @since 1.0
 */
public class Tensor {
    private final float[] vals;
    private final int[] shape;

    /**
     * Length of the data.
     * Equivalent to <code>vals.length</code>.
     * @since 1.0
     */
    public final int length;

    /**
     * Creates a tensor from an array of values and a shape.
     * @param vals A <code>float[]</code> of values. The dimensionality is added by <code>shape</code>.
     * @param shape An <code>int[]</code> of dimensions. This gives <code>vals</code> dimension.
     * @since 1.0
     */
    public Tensor(float[] vals, int[] shape) {
        this.vals = vals;
        this.shape = shape;

        int length = 1;
        for (int i : shape) {
            length *= i;
        }
        this.length = length;
    }

    private Tensor(float[] vals, int[] shape, int length) {
        this.vals = vals;
        this.shape = shape;
        this.length = length;
    }

    /**
     * Returns the shape of the tensor.
     * @return An <code>int[]</code> of the dimensions of the tensor.
     * @since 1.0
     */
    public int[] getShape() {
        return shape;
    }

    /**
     * Returns the value at <code>i</code> based on <code>vals</code>
     * @param i The index to get.
     * @return The value at <code>i</code>.
     * @since 1.0
     */
    public float getVal(int i) {
        return vals[i];
    }

    /**
     * Set the value at <code>i</code> to <code>val</code>.
     * @param i The index to set.
     * @param val The value to set.
     * @since 1.0
     */
    public void setVal(int i, float val) {
        vals[i] = val;
    }

    /**
     * Get the value at a list of indices. This has dimensionality.
     * @param indices The list of indices to get.
     * @return The value at <code>indices</code>.
     * @since 1.0
     */
    public float getVal(int[] indices) {
        int i = 0;
        for (int j = 0; j < indices.length; j++) {
            int iI = indices[j];
            for (int k = indices.length - 1; k > j; k--) {
                iI *= shape[k];
            }
            i += iI;
        }
        return vals[i];
    }

    /**
     * Set the value at a list of indices. This has dimensionality.
     * @param indices The list of indices to set.
     * @param val The value to set.
     * @since 1.0
     */
    public void setVal(int[] indices, float val) {
        int i = 0;
        for (int j = 0; j < indices.length; j++) {
            int iI = indices[j];
            for (int k = indices.length - 1; k > j; k--) {
                iI *= shape[k];
            }
            i += iI;
        }
        vals[i] = val;
    }

    /**
     * Converts a list of indices into a single index.
     * @param indices The list of indices.
     * @return The index.
     * @since 1.0
     */
    public int getIndexFromIndices(int[] indices) {
        int i = 0;
        for (int j = 0; j < indices.length; j++) {
            int iI = indices[j];
            for (int k = indices.length - 1; k > j; k--) {
                iI *= shape[k];
            }
            i += iI;
        }
        return i;
    }

    /**
     * Converts an index into a list of indices.
     * @param i The index.
     * @return The list of indices.
     * @since 1.0
     */
    public int[] getIndicesFromIndex(int i) {
        int[] indices = new int[shape.length];
        for (int j = 0; j < shape.length; j++) {
            int iI = 1;
            for (int k = shape.length - 1; k > j; k--) {
                iI *= shape[k];
            }
            indices[j] = i / iI;
            i -= (i / iI) * iI;
        }
        return indices;
    }

    /**
     * Checks if an object is equal to this tensor.
     * @param o The object to check.
     * @return Whether the object equals this tensor.
     * @since 1.0
     */
    @Override
    public boolean equals(Object o) {
        if (o instanceof Tensor) {
            Tensor other = (Tensor) o;
            if (Arrays.equals(other.shape, this.shape)) {
                if (Arrays.equals(other.vals, this.vals)) {
                    return true;
                }
            }
        }
        return false;
    }

    /**
     * Returns a string summary of the tensor.
     * @return String summary.
     * @since 1.0
     */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("Tensor(Shape=");
        sb.append(Arrays.toString(shape));
        sb.append(", Data=");

        int[] shapeI = new int[shape.length];
        sb.append(new String(new byte[shape.length]).replaceAll("\u0000", "["));
        for (int j = 0; j < length; j++) {
            sb.append(vals[j]);
            int closed = 0;
            for (int k = shape.length - 1; k >= 0; k--) {
                shapeI[k]++;
                if (shapeI[k] == shape[k]) {
                    shapeI[k] = 0;
                    closed++;
                } else {
                    break;
                }
            }
            if (closed == 0) {
                sb.append(", ");
            } else if (j == length - 1) {
                sb.append(new String(new byte[closed]).replaceAll("\u0000", "]"));
            } else {
                sb.append(new String(new byte[closed]).replaceAll("\u0000", "]"));
                sb.append(", ");
                sb.append(new String(new byte[closed]).replaceAll("\u0000", "["));
            }
        }
        sb.append(")");

        return sb.toString();
    }

    /**
     * Create a Tensor from <code>Math.random()</code>
     * @param lb Lower bound of randomness.
     * @param ub Upper bound of randomness.
     * @param shape The shape of the tensor.
     * @return The tensor.
     * @since 1.0
     */
    public static Tensor fromRandom(float lb, float ub, int[] shape) {
        int length = 1;
        for (int i : shape) {
            length *= i;
        }

        float[] vals = new float[length];
        for (int i = 0; i < length; i++) {
            vals[i] = ((float) Math.random()) * (ub - lb) + lb;
        }

        return new Tensor(vals, shape, length);
    }

    /**
     * Create a Tensor from <code>Random</code>
     * @see java.util.Random
     * @param lb Lower bound of randomness.
     * @param ub Upper bound of randomness.
     * @param shape The shape of the tensor.
     * @return The tensor.
     * @since 1.0
     */
    public static Tensor fromRandom(Random random, float lb, float ub, int[] shape) {
        int length = 1;
        for (int i : shape) {
            length *= i;
        }

        float[] vals = new float[length];
        for (int i = 0; i < length; i++) {
            vals[i] = random.nextFloat() * (ub - lb) + lb;
        }

        return new Tensor(vals, shape, length);
    }

}
