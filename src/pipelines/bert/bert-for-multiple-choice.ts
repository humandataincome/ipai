import * as ipdw from "ipdw";
import ort, {InferenceSession} from "../../utils/onnxruntime";
import nj from "numjs";


export class BertForMultipleChoice {
    private session: InferenceSession;

    constructor(session: InferenceSession) {
        this.session = session;
    }

    public static async Load(path: string, progress?: ((progress: number) => void) | undefined): Promise<BertForMultipleChoice> {
        ort.env.wasm.numThreads = 1;

        const persistence = await ipdw.Persistence.getInstance();
        const file = await persistence.fetchOrGet(path, progress);

        const session = await ort.InferenceSession.create(file!, {
            executionProviders: [typeof window === 'undefined' ? "cpu" : "wasm"],
            graphOptimizationLevel: "all",
        });

        return new BertForMultipleChoice(session);
    }

    public async generate(inputIds: number[]): Promise<number[]> {
        const maxLength = 16;
        if (inputIds.length < maxLength)
            inputIds = inputIds.concat(Array(maxLength - inputIds.length).fill(0));
        else if (inputIds.length > maxLength)
            inputIds = inputIds.slice(0, 16)
        const inputIdsTensor = new ort.Tensor("int64", BigInt64Array.from(inputIds.map(v => BigInt(v))), [1, maxLength]);
        const attentionMaskTensor = new ort.Tensor("int64", BigInt64Array.from(inputIds.map(v => BigInt(1 ? v !== 0 : 0))), [1, maxLength]);

        const output = await this.session.run({
            'input_ids': inputIdsTensor,
            'attention_mask': attentionMaskTensor
        });


        let logits = nj.array(output.logits.data as Float32Array, 'float32').reshape(...output.logits.dims);
        logits = logits.flatten();
        logits = nj.sigmoid(logits);

        const res = [];
        for (let i = 0; i < logits.shape[logits.shape.length - 1]; i++)
            if (logits.get(i) > 0.2)
                res.push(i);

        return res;
    }

}
