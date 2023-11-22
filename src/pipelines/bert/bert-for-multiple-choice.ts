import * as ipdw from "ipdw";
import * as tflite from '@tensorflow/tfjs-tflite';


export class BertForMultipleChoice {
    private classifier: tflite.BertNLClassifier;

    constructor(classifier: tflite.BertNLClassifier) {
        this.classifier = classifier;
    }

    public static async Load(path: string, progress?: ((progress: number) => void) | undefined): Promise<BertForMultipleChoice> {
        const persistence = await ipdw.Persistence.getInstance();
        const file = await persistence.fetchOrGet(path, progress);

        const classifier = await tflite.BertNLClassifier.create(file!.buffer);

        return new BertForMultipleChoice(classifier);
    }

    public async generate(input: string): Promise<string> {
        const probabilities = this.classifier.classify(input);
        const max = probabilities.reduce((p, c) => p.probability > c.probability ? p : c);
        return max.className;
    }

}
