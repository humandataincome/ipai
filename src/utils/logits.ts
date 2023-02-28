// Reference https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py

import nj from 'numjs';
import {TensorUtils} from "./tensor";

interface LogitsProcessor {
    call(inputIds: nj.NdArray, scores: nj.NdArray): nj.NdArray;
}

export class TopKLogitsProcessor implements LogitsProcessor {
    private topK: number;
    private readonly filterValue: number;

    constructor(topK: number, filterValue: number = Number.MIN_VALUE) {
        this.topK = topK;
        this.filterValue = filterValue;
    }

    public call(inputIds: nj.NdArray, scores: nj.NdArray): nj.NdArray {
        const topKMin = TensorUtils.topK(scores, this.topK)?.min()!;
        const indicesToRemove = TensorUtils.Where(scores, v => v < topKMin);
        scores = TensorUtils.Fill(scores, indicesToRemove, this.filterValue);
        return scores;
    }
}

export class TemperatureLogitsProcessor implements LogitsProcessor {
    private readonly temperature: number;

    constructor(temperature: number) {
        this.temperature = temperature;
    }

    public call(inputIds: nj.NdArray, scores: nj.NdArray): nj.NdArray {
        return scores.divide(this.temperature);
    }
}

export class TopPLogitsProcessor implements LogitsProcessor {
    private topP: number;
    private readonly filterValue: number;

    constructor(topP: number, filterValue: number = Number.MIN_VALUE) {
        this.topP = topP;
        this.filterValue = filterValue;
    }

    public call(inputIds: nj.NdArray, scores: nj.NdArray): nj.NdArray {
        const [sortedLogits, sortedIndices] = TensorUtils.Sort(scores, false);
        const cumulativeProbs = TensorUtils.Cumsum(sortedLogits);
        const sortedIndicesToRemove = TensorUtils.Where(cumulativeProbs, v => v <= 1 - this.topP);
        const indicesToRemove = TensorUtils.Scatter(sortedIndicesToRemove, 1, sortedIndices, sortedIndicesToRemove);
        scores = TensorUtils.Fill(scores, indicesToRemove, this.filterValue);
        return scores;
    }
}
