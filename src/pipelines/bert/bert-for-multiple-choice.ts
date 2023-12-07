import * as ipdw from "ipdw";
import ort, {InferenceSession} from "../../utils/onnxruntime";
import nj from "numjs";

//const ENVIRONMENT_IS_NODE = typeof process === 'object' && typeof require === 'function';
//const ENVIRONMENT_IS_WEB = typeof window === 'object';
//const ENVIRONMENT_IS_WORKER = typeof importScripts === 'function';
//const ENVIRONMENT_IS_SHELL = !ENVIRONMENT_IS_WEB && !ENVIRONMENT_IS_NODE && !ENVIRONMENT_IS_WORKER;

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

        let logits = nj.array(output.logits.data as Float32Array, 'float32').reshape(...output.logits.dims);
        logits = logits.flatten();
        logits = nj.sigmoid(logits);

        const res = [];
        for (let i = 0; i < logits.shape[logits.shape.length - 1]; i++)
            if (logits.get(i) > 0.15)
                res.push(i);

        return res;
    }

}
