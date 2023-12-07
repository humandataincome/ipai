import * as ipdw from "ipdw";
import ort, {InferenceSession} from "../../utils/onnxruntime";

export class BertForMultipleChoice {
    private session: InferenceSession;

    constructor(session: InferenceSession) {
        this.session = session;
    }

    public static async Load(path: string, progress?: ((progress: number) => void) | undefined): Promise<BertForMultipleChoice> {
        ort.env.wasm.numThreads = 1;

        const persistence = await ipdw.Persistence.getInstance();
        const file = await persistence.fetchOrGet(path, progress);

        const provider = typeof window === 'object' || typeof importScripts === 'function' ? 'wasm' : 'cpu';

        const session = await ort.InferenceSession.create(file!, {
            executionProviders: [provider],
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
        //const attentionMaskTensor = new ort.Tensor("int64", BigInt64Array.from(inputIds.map(v => BigInt(1 ? v !== 0 : 0))), [1, maxLength]);

        //console.log(this.session.inputNames)
        const output = await this.session.run({
            'input_ids': inputIdsTensor,
            //'attention_mask': attentionMaskTensor
        });

        const logits = new Float32Array(output.logits.data.length);

        for (let i = 0; i < output.logits.data.length; i++)
            logits[i] = 1 / (1 + Math.exp(-output.logits.data[i] as number));

        const res = [];
        for (let i = 0; i < logits.length; i++)
            if (logits[i] > 0.15)
                res.push(i);

        return res;
    }

}
