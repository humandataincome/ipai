import ort, {InferenceSession} from "../../utils/onnxruntime"
import * as ipdw from "ipdw";
import nj from 'numjs';
import {TensorUtils} from "../../utils/tensor";
import {TemperatureLogitsProcessor, TopKLogitsProcessor, TopPLogitsProcessor} from "../../utils/logits";

export class GPTNeoXForCausalLm {
    private session: InferenceSession;

    constructor(session: InferenceSession) {
        this.session = session;
    }

    public static async Load(path: string, progress?: ((progress: number) => void) | undefined): Promise<GPTNeoXForCausalLm> {
        ort.env.wasm.numThreads = 1;

        const persistence = await ipdw.Persistence.getInstance();
        const file = await persistence.fetchOrGet(path, progress);

        const session = await ort.InferenceSession.create(file!, {
            executionProviders: [typeof window === 'undefined' ? "cpu" : "wasm"],
            graphOptimizationLevel: "all",
        });

        return new GPTNeoXForCausalLm(session);
    }

    public async generate(
        inputIds: number[],
        doSample = true,
        temperature = 0.9,
        topK = 5,
        topP = 0.9,
        maxLength = 10,
        callback?: (token: number) => Promise<void>
    ): Promise<number[]> {
        const temperatureProcessor = new TemperatureLogitsProcessor(temperature);
        const topKProcessor = new TopKLogitsProcessor(topK);
        const topPProcessor = new TopPLogitsProcessor(topP);

        let allInputIds = inputIds;
        for (let i = 0; i < maxLength; i++) {
            const inputIdsTensor = new ort.Tensor("int64", BigInt64Array.from(allInputIds.map(v => BigInt(v))), [1, allInputIds.length]);
            const attentionMaskTensor = new ort.Tensor("int64", BigInt64Array.from(new Array(allInputIds.length).fill(BigInt(1))), [1, allInputIds.length]);
            const output = await this.session.run({
                'input_ids': inputIdsTensor,
                'attention_mask': attentionMaskTensor
            });

            let logits = nj.array(output.logits.data as Float32Array, 'float32').reshape(...output.logits.dims);
            logits = logits.slice(null, logits.shape[1] - 1, null).flatten();
            logits = temperatureProcessor.call(nj.array(inputIds), logits);
            logits = nj.softmax(logits.subtract(logits.max()));
            logits = topKProcessor.call(nj.array(inputIds), logits);
            logits = topPProcessor.call(nj.array(inputIds), logits);
            let nextTokens: nj.NdArray;
            if (doSample) {
                nextTokens = nj.array(TensorUtils.ArgChoices(logits, logits));
            } else {
                nextTokens = TensorUtils.argmax(logits)!;
            }

            if (callback)
                await callback(nextTokens.get(0));
            allInputIds = [...allInputIds, nextTokens.get(0)];
        }

        return allInputIds;
    }

}
