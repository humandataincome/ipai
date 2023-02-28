import nj from "numjs";

// https://github.com/microsoft/onnxruntime/blob/main/js/common/lib/tensor-impl.ts
// https://github.com/nicolaspanel/numjs/blob/master/src/ndarray.js

export class TensorUtils {
    public static argmax(array: nj.NdArray, axis = -1): nj.NdArray | undefined {
        if (array.shape.length === 0) {
            return undefined;
        }

        let indices;
        if (axis === -1 || axis === array.shape.length - 1) {
            let maxVal = -Infinity;
            let maxIndex = -1;
            for (let i = 0; i < array.shape[array.shape.length - 1]; i++) {
                if (array.get(i) > maxVal) {
                    maxVal = array.get(i);
                    maxIndex = i;
                }
            }
            indices = nj.array([maxIndex]);
        } else {
            throw Error("Not implemented");
        }
        return indices;
    }

    public static argmin(array: nj.NdArray, axis = -1): nj.NdArray | undefined {
        if (array.shape.length === 0) {
            return undefined;
        }

        let indices;
        if (axis === -1 || axis === array.shape.length - 1) {
            let minVal = Infinity;
            let minIndex = -1;
            for (let i = 0; i < array.shape[array.shape.length - 1]; i++) {
                if (array.get(i) < minVal) {
                    minVal = array.get(i);
                    minIndex = i;
                }
            }
            indices = nj.array([minIndex]);
        } else {
            throw Error("Not implemented");
        }
        return indices;
    }

    public static topK(array: nj.NdArray, k: number, axis = -1): nj.NdArray | undefined {
        if (array.shape.length === 0) {
            return undefined;
        }

        let values = array.slice([k]).clone();

        let minIndex = this.argmin(values)?.get(0)!;
        if (axis === -1 || axis === array.shape.length - 1) {
            for (let i = 0; i < array.shape[array.shape.length - 1]; i++) {
                if (array.get(i) > values.get(minIndex)) {
                    values.set(minIndex, array.get(i));
                    minIndex = this.argmin(values)?.get(0)!;
                }
            }
        } else {
            throw Error("Not implemented");
        }
        return values;
    }

    public static Sort(array: nj.NdArray, descending: boolean = true): [nj.NdArray, nj.NdArray] {
        if (array.shape.length !== 1) {
            throw Error("Given array should be one-dimensional");
        }

        let values = array.clone();
        let indices = nj.arange(array.size);
        this.quicksort(values, indices, 0, array.shape[0] - 1);

        if (descending) {
            values = values.slice(null, null, -1);
            indices = indices.slice(null, null, -1);
        }

        return [values.reshape(...array.shape), indices.reshape(...array.shape)];
    }

    public static Where(array: nj.NdArray, cond: (v: number) => boolean): nj.NdArray {
        let res = array.clone();
        for (let i = 0; i < res.shape[res.shape.length - 1]; i++)
            res.set(i, cond(res.get(i)) ? 1 : 0);
        return res;
    }

    public static Fill(array: nj.NdArray, where: nj.NdArray, value: number): nj.NdArray {
        let res = array.clone();
        for (let i = 0; i < res.shape[res.shape.length - 1]; i++)
            if (where.get(i))
                res.set(i, value);
        return res;
    }

    public static Cumsum(array: nj.NdArray<number>): nj.NdArray {
        if (array.shape.length !== 1) {
            throw Error("Given array should be one-dimensional");
        }

        let cumulative = nj.zeros(array.shape);
        let total = 0;
        for (let i = 0; i < array.size; i++) {
            total += array.get(i);
            cumulative.set(i, total);
        }

        return cumulative;
    }

    public static Scatter(self: nj.NdArray, dim: number, index: nj.NdArray, src: nj.NdArray): nj.NdArray {
        if (index.shape.join(',') !== src.shape.join(',')) {
            throw new Error('index and src should have the same shape');
        }
        if (dim < 0 || dim >= self.shape.length) {
            throw new Error('dim should be between 0 and the rank of the array');
        }

        const shape = self.shape;
        const strides = self.stride;
        const offset = self.offset;

        const flatIndex = index.flatten<number>();
        const flatSrc = src.flatten<number>();
        const numElements = flatIndex.size;

        for (let i = 0; i < numElements; i++) {
            const idx = flatIndex.get(i);
            const srcVal = flatSrc.get(i);

            if (idx >= shape[dim]) {
                throw new Error('index is out of bounds for dimension ' + dim);
            }

            let newOffset = offset;
            for (let j = 0; j < shape.length; j++) {
                if (j === dim) {
                    newOffset += idx * strides[dim];
                } else {
                    const coord = Math.floor((offset / strides[j]) % shape[j]);
                    newOffset += coord * strides[j];
                }
            }

            self.set(srcVal, newOffset);
        }

        return self;
    }

    public static assignAt(self: nj.NdArray, index: nj.NdArray, src: nj.NdArray): nj.NdArray {
        const flatIndex = index.flatten<number>();
        const flatSrc = src.flatten<number>();
        const numElements = flatIndex.size;

        for (let i = 0; i < numElements; i++) {
            const index = flatIndex.get(i);
            const value = flatSrc.get(i);
            self.set(index, value);
        }

        return self;
    }

    public static ArgChoices<T>(population: nj.NdArray<T>, weights?: nj.NdArray<number>, k: number = 1): number[] {
        const result: number[] = [];
        const cumulativeWeights: number[] = [];
        let totalWeight = 0;

        // If weights are not specified, assume equal weighting of elements in the population
        if (weights === undefined) {
            weights = nj.ones(population.size);
        }

        // Flatten the weights array, if it is not already flattened
        if (weights.ndim > 1) {
            weights = weights.flatten();
        }

        // Calculate the cumulative weights of the elements in the population
        for (let i = 0; i < population.size; i++) {
            totalWeight += weights.get(i);
            cumulativeWeights[i] = totalWeight;
        }

        // Select k elements from the population with replacement based on the weights
        for (let i = 0; i < k; i++) {
            const rand = Math.random() * totalWeight;
            let index = 0;

            // Find the index of the first element with a cumulative weight greater than the random value
            while (cumulativeWeights[index] < rand) {
                index++;
            }

            // Add the selected element to the result
            result.push(index);
        }

        return result;
    }

    private static quicksort(values: nj.NdArray, indices: nj.NdArray, low: number, high: number) {
        if (low < high) {
            const pivotIndex = this.partition(values, indices, low, high);
            this.quicksort(values, indices, low, pivotIndex - 1);
            this.quicksort(values, indices, pivotIndex + 1, high);
        }
    }

    private static partition(values: nj.NdArray, indices: nj.NdArray, low: number, high: number): number {
        const pivot = values.get(high);
        let i = low - 1;

        for (let j = low; j < high; j++) {
            if (values.get(j) < pivot) {
                i++;
                const tempValue = values.get(i);
                const tempIndex = indices.get(i);
                values.set(i, values.get(j));
                indices.set(i, indices.get(j));
                values.set(j, tempValue);
                indices.set(j, tempIndex);
            }
        }

        const tempValue = values.get(i + 1);
        const tempIndex = indices.get(i + 1);
        values.set(i + 1, values.get(high));
        indices.set(i + 1, indices.get(high));
        values.set(high, tempValue);
        indices.set(high, tempIndex);

        return i + 1;
    }


}
