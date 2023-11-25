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

        const session = await ort.InferenceSession.create(file!, {
            executionProviders: [typeof window === 'undefined' ? "cpu" : "wasm"],
            graphOptimizationLevel: "all",
        });

        return new BertForMultipleChoice(session);
    }

    public async generate(input: string): Promise<number> {
        const probabilities = [0.8, 0.5]
        const max = probabilities.reduce((p, c) => p > c ? p : c);
        return max;
    }

}
