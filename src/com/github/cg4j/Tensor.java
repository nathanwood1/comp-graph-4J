package com.github.cg4j;

import com.github.cg4j.exception.IllegalShapeException;

import java.util.Arrays;
import java.util.Random;

/**
 * The Tensor class represents an N-Dimensional Array, also called a tensor.
 * @since 1.0
 * @author nathanwood1
 */
public class Tensor {
    private final float[] vals;
    private final int[] shape;

    /**
     * Length of the data.
     * Equivalent to {@code vals.length}.
     * @since 1.0
     */
    public final int length;

    /**
     * Creates a tensor from an array of values and a shape.
     * @param vals A {@code float[]} of values. The dimensionality is added by {@code shape}.
     * @param shape An {@code int[]} of dimensions. This gives {@code vals} dimension.
     * @throws IllegalShapeException if the given shape contains any values <= 0.
     * @throws IllegalShapeException if the length of {@code vals} doesn't equal the length given by {@code shape}
     * @since 1.0
     */
    public Tensor(float[] vals, int[] shape) {
        this.vals = vals;
        this.shape = shape;

        int length = 1;
        for (int x : shape) {
            if (x < 0) {
                throw new IllegalShapeException(
                        "Shape cannot have any negative values "
                                + Arrays.toString(shape)
                );
            }
            if (x == 0) {
                throw new IllegalShapeException(
                        "Shape cannot have any zeros "
                                + Arrays.toString(shape)
                );
            }
            length *= x;
        }
        this.length = length;

        if (vals.length != length) {
            throw new IllegalShapeException(
                    "Shape doesn't match length of vals ("
                    + length
                    + " != "
                    + vals.length
                    + ")"
            );
        }
    }

    private Tensor(float[] vals, int[] shape, int length) {
        this.vals = vals;
        this.shape = shape;
        this.length = length;
    }

    /**
     * Returns the shape of the tensor.
     * @return An {@code int[]} of the dimensions of the tensor.
     * @since 1.0
     */
    public int[] getShape() {
        return shape;
    }

    /**
     * Returns the value at {@code i} based on {@code vals}
     * @param i The index to get.
     * @return The value at {@code i}.
     * @since 1.0
     */
    public float getVal(int i) {
        return vals[i];
    }

    /**
     * Set the value at {@code i} to {@code val}.
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
     * @return The value at {@code indices}.
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
     * Same as toString() but doesn't display 'Tensor(Shape=...'
     * @return String summary.
     * @since 1.0
     */
    public String arrayToString() {
        StringBuilder sb = new StringBuilder();

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

        return sb.toString();
    }

    /**
     * Create a Tensor from {@code Math.random()}
     * @param lb Lower bound of randomness.
     * @param ub Upper bound of randomness.
     * @param shape The shape of the tensor.
     * @return The tensor.
     * @since 1.0
     */
    public static Tensor fromRandom(float lb, float ub, int[] shape) {
        int length = 1;
        for (int x : shape) {
            length *= x;
        }

        float[] vals = new float[length];
        for (int i = 0; i < length; i++) {
            vals[i] = ((float) Math.random()) * (ub - lb) + lb;
        }

        return new Tensor(vals, shape, length);
    }

    /**
     * Create a Tensor from {@code Random}
     * @see java.util.Random
     * @param lb Lower bound of randomness.
     * @param ub Upper bound of randomness.
     * @param shape The shape of the tensor.
     * @return The tensor.
     * @since 1.0
     */
    public static Tensor fromRandom(Random random, float lb, float ub, int[] shape) {
        int length = 1;
        for (int x : shape) {
            length *= x;
        }

        float[] vals = new float[length];
        for (int i = 0; i < length; i++) {
            vals[i] = random.nextFloat() * (ub - lb) + lb;
        }

        return new Tensor(vals, shape, length);
    }

}
